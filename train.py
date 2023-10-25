import logging
import multiprocessing as mp
import os
import pickle
from functools import partial

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import datasets
import hydra
import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.sharding as shd
import numpy as np
import optax
import tqdm
import transformers
import wandb
from flax.core.frozen_dict import freeze, unfreeze
from flax.training.early_stopping import EarlyStopping
from omegaconf import OmegaConf

import lmrax.optimizers
from lmrax.datasets.utils import seed_worker, preprocess_function, data_loader
from lmrax.dir_manager import DirManager
from lmrax.sharding import get_batch_shardings, get_params_shardings

datasets.disable_caching()


def predict_fn(params, batch, model, rng=None):
    if rng is None:
        training = False
        encoder_rng, chosen_rng, rejected_rng = None, None, None
    else:
        training = True
        encoder_rng, chosen_rng, rejected_rng = jax.random.split(rng, 3)

    encoder_outputs = model.encode(
        params=params,
        input_ids=batch["context_input_ids"],
        attention_mask=batch["context_attention_mask"],
        train=training,
        dropout_rng=encoder_rng,
    )

    chosen_reward = (
        model.decode(
            params=params,
            encoder_outputs=encoder_outputs,
            encoder_attention_mask=batch["context_attention_mask"],
            decoder_input_ids=batch["chosen_input_ids"],
            decoder_attention_mask=batch["chosen_attention_mask"],
            train=training,
            dropout_rng=chosen_rng,
        ).last_hidden_state
        @ params["reward_head"]
    )

    rejected_reward = (
        model.decode(
            params=params,
            encoder_outputs=encoder_outputs,
            encoder_attention_mask=batch["context_attention_mask"],
            decoder_input_ids=batch["rejected_input_ids"],
            decoder_attention_mask=batch["rejected_attention_mask"],
            train=training,
            dropout_rng=rejected_rng,
        ).last_hidden_state
        @ params["reward_head"]
    )

    chosen_reward = jnp.tanh(chosen_reward)  # (B, L)
    rejected_reward = jnp.tanh(rejected_reward)  # (B, L)

    # mask out paddings
    chosen_reward = jnp.where(batch["chosen_attention_mask"] == 0, 0.0, chosen_reward)  # (B, L)

    rejected_reward = jnp.where(batch["rejected_attention_mask"] == 0, 0.0, rejected_reward)  # (B, L)

    chosen_score = jnp.sum(chosen_reward, axis=-1)  # (B,)
    rejected_score = jnp.sum(rejected_reward, axis=-1)  # (B,)

    log_prob_chosen = jax.nn.log_sigmoid(chosen_score - rejected_score)  # (B,)
    log_prob_rejected = jax.nn.log_sigmoid(rejected_score - chosen_score)  # (B,)

    return log_prob_chosen, log_prob_rejected


def loss_fn(params, batch, dropout_rng, model):
    weight = batch["weight"]
    log_prob_chosen, log_prob_rejected = predict_fn(params, batch, model, dropout_rng)
    loss = -jnp.mean(weight * log_prob_chosen + (1 - weight) * log_prob_rejected)
    acc = jnp.mean(log_prob_chosen > log_prob_rejected)

    return loss, acc


def grad_fn(params, batch, rng, model):
    return jax.value_and_grad(loss_fn, has_aux=True)(params, batch, rng, model)


def _update_fn(model, optimizer, rng, batch, params, state, cfg=None):
    half_params = jax.tree_map(lambda x: x.astype(jnp.bfloat16), params)
    updates = jax.tree_map(jnp.zeros_like, half_params)

    def _inner_update_fn(i, data):
        _loss, _acc, _grads, _rng = data
        _, _rng = jax.random.split(_rng)
        _batch = batch_select(batch, i, 1)
        (__loss, __acc), __grads = grad_fn(half_params, _batch, _rng, model)
        _loss = _loss + __loss
        _acc = _acc + __acc
        _grads = jax.tree_map(lambda x, y: x + y, _grads, __grads)
        return _loss, _acc, _grads, _rng

    loss, acc, updates, rng = jax.lax.fori_loop(
        0,
        cfg.gradient_accumulation,
        _inner_update_fn,
        (0.0, 0.0, updates, rng),
    )

    loss = loss.astype(jnp.float32) / cfg.gradient_accumulation
    acc = acc.astype(jnp.float32) / cfg.gradient_accumulation
    grad_norm = grad_norm_fn(updates).astype(jnp.float32) / cfg.gradient_accumulation

    updates = jax.tree_map(lambda x: x.astype(jnp.float32) / cfg.gradient_accumulation, updates)

    updates, state = optimizer.update(updates, state, params)
    params = optax.apply_updates(params, updates)
    return loss, acc, params, state, grad_norm, rng


def batch_select(batch, idx, axis=0):
    return jax.tree_map(lambda x: jnp.take(x, idx, axis=axis), batch)


def _eval_fn(params, batch, model, cfg):
    params = jax.tree_map(lambda x: x.astype(jnp.bfloat16), params)
    bs = batch["weight"].shape[0]
    loss, acc = loss_fn(params, batch, None, model)
    loss = loss.astype(jnp.float32)
    acc = acc.astype(jnp.float32)
    return loss * bs, acc * bs


def mean_grads_fn(grads):
    return jax.tree_map(lambda x: jnp.mean(x, axis=0), grads)


def grad_norm_fn(grads):
    return jnp.sqrt(
        jax.tree_util.tree_reduce(
            lambda x, y: x + y,
            jax.tree_map(lambda x: jnp.linalg.norm(x) ** 2, grads),
        )
    )


@partial(jax.jit, static_argnums=(1,))
def batch_reshape(batch, d0):
    return jax.tree_map(
        lambda x: x.reshape(d0, -1, *x.shape[1:]),
        batch,
    )


class Trainer:
    def __init__(
        self,
        cfg,
        train_ds,
        val_ds,
        update_fn,
        eval_fn,
    ):
        self.logger = logging.getLogger(__name__)
        self.cfg = cfg
        self.train_ds = train_ds
        self.val_ds = val_ds

        self.steps = 0
        self.epoch = 0
        self.params_updates = 0
        self.batch_size = cfg.batch_size_per_device * cfg.num_dp_devices * cfg.gradient_accumulation
        self.eval_batch_size = cfg.batch_size_per_device * cfg.num_dp_devices
        self.max_length = cfg.max_length

        self.es = EarlyStopping(
            patience=cfg.patience,
        )
        self.rng = jax.tree_map(np.asarray, jax.random.PRNGKey(cfg.seed))
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.model_name)

        scheduler_cfg = OmegaConf.to_object(cfg.scheduler)
        optimizer_cfg = OmegaConf.to_object(cfg.optimizer)
        scheduler_cls = lmrax.optimizers.get_scheduler(scheduler_cfg.pop("name"))

        self.scheduler = scheduler_cls(**scheduler_cfg)

        optimizer_cls = lmrax.optimizers.get_optimizer(optimizer_cfg.pop("name"))
        optimizer_chains = [
            optimizer_cls(self.scheduler, **optimizer_cfg),
        ]
        if cfg.max_grad_norm is not None:
            optimizer_chains.append(optax.clip_by_global_norm(cfg.max_grad_norm))
        elif cfg.max_grad_value is not None:
            optimizer_chains.append(optax.clip(cfg.max_grad_value))
        self.optimizer = optax.chain(*optimizer_chains)

        devices = np.array(jax.devices()).reshape(cfg.num_dp_devices, cfg.num_tp_devices)

        # dp: data parallel, tp: tensor parallel
        self.mesh = shd.Mesh(devices, ("dp", "tp"))

        self.train_loader = self.get_dataloader(self.train_ds, drop_last=True)
        self.val_loader = self.get_dataloader(self.val_ds, batch_size=self.eval_batch_size, shuffle=False)

        self.dmgr = DirManager(cfg.save_dir, maximize=True)

        if cfg.override:
            self.logger.warning(
                f"Overriding previous checkpoint under {cfg.save_dir}\n" + f"Loading model from f{cfg.model_name}"
            )
            path = self.cfg.model_name
        elif self.dmgr.last_model is not None:
            self.logger.info(f"Loading previous checkpoint from {self.dmgr.last_model}")
            path = self.dmgr.last_model
        else:
            self.logger.info(f"Loading model from {self.cfg.model_name}")
            path = self.cfg.model_name

        self.model, params = transformers.FlaxAutoModel.from_pretrained(path, _do_init=False, dtype=jnp.bfloat16)

        batch = next(iter(self.train_loader))

        params = self.model.init_weights(self.rng, (cfg.batch_size_per_device, cfg.max_length), params)
        if params.get("reward_head", None) is None:
            params = unfreeze(params)
            ndim = self.model.config.d_model
            params["reward_head"] = jax.random.normal(self.rng, (ndim,)) / ndim
            params = freeze(params)
        params = jax.tree_map(np.asarray, params)

        none_shd = shd.NamedSharding(self.mesh, shd.PartitionSpec())

        params_shardings = freeze(get_params_shardings(self.mesh, params))
        self.params = jax.device_put(params, params_shardings)
        if cfg.reset_status or cfg.override:
            self.logger.warning(f"Ignoring saved status under f{cfg.save_dir}")
            self.state = self.optimizer.init(self.params)
        elif self.dmgr.last_model is not None:
            self.logger.info(f"Loading model from f{self.dmgr.last_model}")
            self.load_status(self.dmgr.last_model)
        else:
            self.state = self.optimizer.init(self.params)

        batch_shardings = get_batch_shardings(self.mesh, batch)

        # TODO(yongchanghao): this is a hack
        def get_state_shardings(x):
            x = unfreeze(x)
            if isinstance(x, dict):
                return params_shardings
            return shd.NamedSharding(self.mesh, shd.PartitionSpec())

        state_shardings = jax.tree_util.tree_map(
            get_state_shardings,
            self.state,
            is_leaf=lambda x: isinstance(unfreeze(x), (dict, optax.EmptyState)),
        )

        self.state = jax.device_put(self.state, state_shardings)

        def wrapped_update_fn(rng, batch, params, state):
            return update_fn(
                self.model,
                self.optimizer,
                rng,
                batch,
                params,
                state,
                cfg,
            )

        def wrapped_eval_fn(params, batch):
            return eval_fn(
                params,
                batch,
                self.model,
                cfg,
            )

        self.update_fn = jax.jit(
            wrapped_update_fn,
            in_shardings=(
                none_shd,  # rng
                batch_shardings,  # batch
                params_shardings,  # params
                state_shardings,  # state
            ),
            out_shardings=(
                none_shd,  # loss
                none_shd,  # acc
                params_shardings,  # params
                state_shardings,  # state
                none_shd,  # grad_norm
                none_shd,  # rng
            ),
        )

        self.eval_fn = jax.jit(
            wrapped_eval_fn,
            in_shardings=(params_shardings, batch_shardings),
            out_shardings=(none_shd, none_shd),
        )

    def get_dataloader(self, ds, batch_size=None, shuffle=False, drop_last=True):
        batch_size = batch_size or self.batch_size
        if drop_last:
            self.epoch_length = len(ds) // batch_size
        else:
            self.epoch_length = (len(ds) + batch_size - 1) // batch_size
        return data_loader(
            rng=self.rng,
            dataset=ds,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def train_epoch(self):
        with tqdm.tqdm(range(self.epoch_length), desc=f"Epoch {self.epoch}", position=self.steps) as bar:
            for _ in range(self.steps):
                next(self.train_loader)
            for _ in range(self.steps, self.epoch_length):
                batch = next(self.train_loader)
                self.steps += 1
                self.params_updates += 1
                batch = batch_reshape(
                    batch,
                    self.cfg.batch_size_per_device * self.cfg.num_dp_devices,
                )
                # _, self.rng = jax.random.split(self.rng)
                (
                    loss,
                    acc,
                    self.params,
                    self.state,
                    grad_norm,
                    self.rng,
                ) = self.update_fn(self.rng, batch, self.params, self.state)

                post_fix = {
                    "loss": jax.device_get(loss).mean(),
                    "grad_norm": jax.device_get(grad_norm).mean(),
                    "acc": jax.device_get(acc).mean(),
                    "steps": self.params_updates,
                    "lr": jax.device_get(self.scheduler(self.params_updates)),
                }
                bar.set_postfix(post_fix)

                if self.steps % self.epoch_length == 0:
                    self.epoch += 1
                    self.steps = 0

                wandb.log(
                    {"train/" + k: v for k, v in post_fix.items()},
                    step=self.params_updates,
                )
                if self.params_updates % self.cfg.eval_steps == 0:
                    results = self.evaluate()
                    wandb.log(
                        {"val/" + k: v for k, v in results.items()},
                        step=self.params_updates,
                    )
                    improved, self.es = self.es.update(-results["acc"])
                    if self.es.should_stop:
                        return True
                    self.save(
                        f"model_best_acc_{results['acc']:.4f}",
                    )
                if self.params_updates % self.cfg.save_steps == 0:
                    self.save(f"model_{self.params_updates}")
                if self.params_updates >= self.cfg.max_updates:
                    return True
                bar.update()
        return False

    def save(self, name):
        def cast_to_fp32(param):
            if isinstance(param, jnp.ndarray) and jnp.issubdtype(param.dtype, jnp.floating):
                param = param.astype(jnp.float32)
            return np.asarray(param)

        self.model.save_pretrained(
            os.path.join(self.cfg.save_dir, name),
            params=self.params,
            push_to_hub=False,
        )
        self.tokenizer.save_pretrained(
            os.path.join(self.cfg.save_dir, name),
            push_to_hub=False,
        )
        status_dict = {
            "epoch": self.epoch,
            "steps": self.steps,
            "params_updates": self.params_updates,
            "es": self.es,
            "state": jax.tree_map(cast_to_fp32, self.state),
        }
        status_path = os.path.join(self.cfg.save_dir, name, "status.pkl")
        with open(status_path, "wb") as f:
            pickle.dump(status_dict, f)

        if os.path.lexists(os.path.join(self.cfg.save_dir, "model_last")):
            os.unlink(os.path.join(self.cfg.save_dir, "model_last"))
        os.symlink(
            os.path.abspath(os.path.join(self.cfg.save_dir, name)),
            os.path.join(self.cfg.save_dir, "model_last"),
        )
        self.dmgr.purge_old(self.cfg.keep_last)
        self.dmgr.purge_worse(self.cfg.keep_best)

    def load_status(self, name):
        status_path = os.path.join(self.cfg.save_dir, name, "status.pkl")
        with open(status_path, "rb") as f:
            status_dict = pickle.load(f)
        self.epoch = status_dict["epoch"]
        self.steps = status_dict["steps"]
        self.params_updates = status_dict["params_updates"]
        self.es = status_dict["es"]
        self.state = status_dict["state"]

    def train(self):
        for i in range(self.cfg.max_epochs):
            finished = self.train_epoch()
            if finished:
                break

    def evaluate(self):
        avg_loss = 0.0
        avg_acc = 0.0
        with tqdm.tqdm(self.val_loader, desc="Evaluating") as bar:
            for batch in bar:
                loss, acc = self.eval_fn(self.params, batch)
                loss = jax.device_get(loss)
                acc = jax.device_get(acc)
                avg_loss += loss
                avg_acc += acc

                bar.set_postfix(
                    {
                        "loss": loss / self.eval_batch_size,
                        "acc": acc / self.eval_batch_size,
                    }
                )

        avg_loss /= len(self.val_ds)
        avg_acc /= len(self.val_ds)

        return {
            "loss": avg_loss,
            "acc": avg_acc,
            "steps": self.params_updates,
        }


@hydra.main(version_base=None, config_path="config", config_name="tp")
def main(cfg):
    seed_worker(cfg.seed)
    train_ds = datasets.load_dataset(cfg.dataset.name, split=cfg.dataset.train)
    ori_train_len = len(train_ds)
    train_ds = train_ds.filter(
        lmrax.datasets.utils.get_filter_fn(cfg),
        num_proc=mp.cpu_count(),
    )
    # train_ds = train_ds.map(
    #     lmrax.datasets.utils.get_map_fn(cfg),
    #     remove_columns=train_ds.features.keys(),
    #     num_proc=mp.cpu_count(),
    #     load_from_cache_file=False,
    # )

    def wrapped_preprocess_function(examples):
        return preprocess_function(
            examples,
            cfg,
            tokenizer=transformers.AutoTokenizer.from_pretrained(cfg.model_name),
            max_source_length=cfg.max_length,
            max_target_length=cfg.max_length,
        )

    train_ds = train_ds.map(
        wrapped_preprocess_function,
        batched=True,
        num_proc=mp.cpu_count(),
        load_from_cache_file=False,
        remove_columns=train_ds.features.keys(),
    )

    val_ds = datasets.load_dataset(cfg.dataset.name, split=cfg.dataset.val)
    ori_val_len = len(val_ds)
    val_ds = val_ds.map(
        lmrax.datasets.utils.get_map_fn(cfg),
        remove_columns=val_ds.features.keys(),
        batched=True,
        num_proc=mp.cpu_count(),
        load_from_cache_file=False,
    )

    trainer = Trainer(
        cfg=cfg,
        train_ds=train_ds,
        val_ds=val_ds,
        update_fn=_update_fn,
        eval_fn=_eval_fn,
    )
    num_params = jax.tree_util.tree_reduce(
        lambda x, y: x + y,
        jax.tree_map(lambda x: x.size, trainer.params),
    )
    wandb.init(project="pf", config=OmegaConf.to_object(cfg), dir=cfg.save_dir)
    wandb.define_metric("val/loss", summary="min")
    wandb.define_metric("val/acc", summary="max")
    wandb.run.config["ori_train_size"] = ori_train_len
    wandb.run.config["ori_val_size"] = ori_val_len
    wandb.run.config["train_size"] = len(train_ds)
    wandb.run.config["val_size"] = len(val_ds)
    wandb.run.config["num_params"] = format(num_params, ",")

    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    """
    Alternatively, use ```
    jax.config.update("jax_default_prng_impl", "rbg")
    ``` to reduce the communication overhead.
    """
    jax.config.update("jax_threefry_partitionable", True)
    main()
