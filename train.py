import multiprocessing as mp
import os
from functools import partial

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
from jax.experimental.pjit import pjit
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import lmrax.optimizers
from lmrax.datasets.preference_feedback import FlaxDataCollatorForSeq2SeqPF
from lmrax.datasets.utils import seed_worker
from lmrax.early_stopping import EarlyStopping, EarlyStoppingMode
from lmrax.sharding import get_batch_shardings, get_params_shardings


def predict_fn(params, batch, model, rng=None):
    if rng is None:
        training = False
        encoder_rng, chosen_rng, rejected_rng = None, None, None
    else:
        training = True
        encoder_rng, chosen_rng, rejected_rng = jax.random.split(rng, 3)
    context = batch["context"]
    chosen = batch["chosen"]
    rejected = batch["rejected"]

    encoder_outputs = model.encode(
        params=params,
        input_ids=context["input_ids"],
        attention_mask=context["attention_mask"],
        train=training,
        dropout_rng=encoder_rng,
    )

    chosen_reward = model.decode(
        params=params,
        encoder_outputs=encoder_outputs,
        encoder_attention_mask=context["attention_mask"],
        decoder_input_ids=chosen["input_ids"],
        decoder_attention_mask=chosen["attention_mask"],
        train=training,
        dropout_rng=chosen_rng,
    ).last_hidden_state.mean(axis=-1)

    rejected_reward = model.decode(
        params=params,
        encoder_outputs=encoder_outputs,
        encoder_attention_mask=context["attention_mask"],
        decoder_input_ids=rejected["input_ids"],
        decoder_attention_mask=rejected["attention_mask"],
        train=training,
        dropout_rng=rejected_rng,
    ).last_hidden_state.mean(axis=-1)

    chosen_reward = jnp.tanh(chosen_reward)  # (B, L)
    rejected_reward = jnp.tanh(rejected_reward)  # (B, L)

    # mask out paddings
    chosen_reward = jnp.where(
        chosen["attention_mask"] == 0, 0.0, chosen_reward
    )  # (B, L)

    rejected_reward = jnp.where(
        rejected["attention_mask"] == 0, 0.0, rejected_reward
    )  # (B, L)

    chosen_score = jnp.sum(chosen_reward, axis=-1)  # (B,)
    rejected_score = jnp.sum(rejected_reward, axis=-1)  # (B,)

    log_prob_chosen = jax.nn.log_sigmoid(chosen_score - rejected_score)  # (B,)
    log_prob_rejected = jax.nn.log_sigmoid(
        rejected_score - chosen_score
    )  # (B,)

    return log_prob_chosen, log_prob_rejected


def loss_fn(params, batch, dropout_rng, model):
    weight = batch["weight"]
    log_prob_chosen, log_prob_rejected = predict_fn(
        params, batch, model, dropout_rng
    )
    loss = -jnp.mean(
        weight * log_prob_chosen + (1 - weight) * log_prob_rejected
    )
    acc = jnp.mean(log_prob_chosen > log_prob_rejected)

    return loss, acc


def grad_fn(params, batch, rng, model):
    return jax.value_and_grad(loss_fn, has_aux=True)(params, batch, rng, model)


def _update_fn(model, optimizer, rng, batch, params, state, cfg=None):
    params = jax.tree_map(lambda x: x.astype(jnp.bfloat16), params)
    grads = jax.tree_map(jnp.zeros_like, params)
    loss = 0.0
    acc = 0.0

    def _inner_update_fn(i, data):
        _loss, _acc, _grads, _rng = data
        _, _rng = jax.random.split(_rng)
        _batch = batch_select(batch, i, 1)
        (__loss, __acc), __grads = grad_fn(params, _batch, _rng, model)
        _loss = _loss + __loss
        _acc = _acc + __acc
        _grads = jax.tree_map(lambda x, y: x + y, _grads, __grads)
        return _loss, _acc, _grads, _rng

    loss, acc, grads, rng = jax.lax.fori_loop(
        0, cfg.gradient_accumulation, _inner_update_fn, (loss, acc, grads, rng)
    )

    loss = loss.astype(jnp.float32) / cfg.gradient_accumulation
    grads = jax.tree_map(
        lambda x: x.astype(jnp.float32) / cfg.gradient_accumulation, grads
    )
    acc = acc.astype(jnp.float32) / cfg.gradient_accumulation
    grad_norm = grad_norm_fn(grads)

    params = jax.tree_map(lambda x: x.astype(jnp.float32), params)
    updates, state = optimizer.update(grads, state, params)
    params = optax.apply_updates(params, updates)
    return loss, acc, params, state, grad_norm, rng


def batch_select(batch, idx, axis=0):
    return jax.tree_map(lambda x: jnp.take(x, idx, axis=axis), batch)


def _eval_fn(params, batch, model, cfg):
    bs = batch["weight"].shape[0]
    params = jax.tree_map(lambda x: x.astype(jnp.bfloat16), params)
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
    def __init__(self, cfg, model, tokenizer, train_ds, val_ds, optimizer):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.optimizer = optimizer
        self.steps = 0
        self.epoch = 0
        self.params_updates = 0
        self.batch_size = (
            cfg.batch_size_per_device
            * cfg.num_dp_devices
            * cfg.gradient_accumulation
        )
        self.eval_batch_size = cfg.batch_size_per_device * cfg.num_dp_devices
        self.max_length = cfg.max_length

        self.params_shardings = None
        self.state_shardings = None

        self.early_stopping = EarlyStopping(
            patience=cfg.patience,
            maximize=True,
        )

        devices = np.array(jax.devices()).reshape(
            cfg.num_dp_devices, cfg.num_tp_devices
        )

        # dp: data parallel, tp: tensor parallel
        self.mesh = shd.Mesh(devices, ("dp", "tp"))

        self.train_loader = self.get_dataloader(self.train_ds, drop_last=True)
        self.val_loader = self.get_dataloader(
            self.val_ds, batch_size=self.eval_batch_size, shuffle=False
        )

        self.update_fn = None
        self.eval_fn = None

    def get_data_collator(self):
        return FlaxDataCollatorForSeq2SeqPF(
            tokenizer=self.tokenizer,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

    def get_dataloader(
        self, ds, batch_size=None, shuffle=True, drop_last=False
    ):
        return DataLoader(
            ds,
            batch_size=batch_size or self.batch_size,
            collate_fn=self.get_data_collator(),
            pin_memory=False,
            drop_last=drop_last,
            worker_init_fn=seed_worker,
            shuffle=shuffle,
        )

    def train_epoch(self, params, state, rng):
        with tqdm.tqdm(self.train_loader, desc=f"Epoch {self.epoch}") as bar:
            for batch in bar:

                self.steps += 1
                batch = batch_reshape(
                    batch,
                    self.cfg.batch_size_per_device * self.cfg.num_dp_devices,
                )

                loss, acc, params, state, grad_norm, rng = self.update_fn(
                    rng, batch, params, state
                )

                post_fix = {
                    "loss": jax.device_get(loss).mean(),
                    "grad_norm": jax.device_get(grad_norm).mean(),
                    "acc": jax.device_get(acc).mean(),
                    "steps": self.params_updates,
                }
                bar.set_postfix(post_fix)

                self.params_updates += 1
                wandb.log(
                    {"train/" + k: v for k, v in post_fix.items()},
                    step=self.params_updates,
                )

                if self.params_updates % self.cfg.save_steps == 0:
                    self.save(params, f"model_{self.params_updates}")
                if self.params_updates % self.cfg.eval_steps == 0:
                    results = self.evaluate(params)
                    wandb.log(
                        {"val/" + k: v for k, v in results.items()},
                        step=self.params_updates,
                    )
                    es_mode = self.early_stopping(results["acc"])
                    if es_mode == EarlyStoppingMode.STOP:
                        return True, params, state, rng
                    elif es_mode == EarlyStoppingMode.BEST:
                        self.save(
                            params,
                            f"model_best_acc_{results['acc']:.4f}",
                        )
                if self.params_updates >= self.cfg.max_updates:
                    return True, params, state, rng

        return False, params, state, rng

    def init(self, params):
        batch = next(iter(self.train_loader))

        params = jax.tree_map(np.asarray, params)
        params_shardings = freeze(get_params_shardings(self.mesh, params))
        batch_shardings = get_batch_shardings(self.mesh, batch)

        state = self.optimizer.init(params)

        # TODO(yongchanghao): this is a hack
        def get_state_shardings(x):
            x = unfreeze(x)
            if isinstance(x, dict):
                return params_shardings
            return shd.NamedSharding(self.mesh, shd.PartitionSpec())

        state_shardings = jax.tree_util.tree_map(
            get_state_shardings,
            state,
            is_leaf=lambda x: isinstance(
                unfreeze(x), (dict, optax.EmptyState)
            ),
        )

        def wrapped_update_fn(rng, batch, params, state):
            return _update_fn(
                self.model,
                self.optimizer,
                rng,
                batch,
                params,
                state,
                self.cfg,
            )

        none_shd = shd.NamedSharding(self.mesh, shd.PartitionSpec())

        self.update_fn = pjit(
            wrapped_update_fn,
            in_axis_resources=(
                none_shd,  # rng
                batch_shardings,  # batch
                params_shardings,  # params
                state_shardings,  # state
            ),
            out_axis_resources=(
                none_shd,  # loss
                none_shd,  # acc
                params_shardings,  # params
                state_shardings,  # state
                none_shd,  # grad_norm
                none_shd,  # rng
            ),
            donate_argnums=(2, 3),
        )

        def wrapped_eval_fn(params, batch):
            return _eval_fn(params, batch, self.model, self.cfg)

        self.eval_fn = pjit(
            wrapped_eval_fn,
            in_axis_resources=(params_shardings, batch_shardings),
            out_axis_resources=(none_shd, none_shd),
        )

        return params, state

    def save(self, params, name):
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        self.model.save_pretrained(
            os.path.join(self.cfg.save_dir, name),
            params=params,
            push_to_hub=False,
        )
        self.tokenizer.save_pretrained(
            os.path.join(self.cfg.save_dir, name),
            push_to_hub=False,
        )

    def train(self, params, state, rng):
        for i in range(self.cfg.max_epochs):
            self.epoch += 1
            finished, params, state, rng = self.train_epoch(params, state, rng)
            if finished:
                break

    def evaluate(self, params):
        avg_loss = 0.0
        avg_acc = 0.0
        with tqdm.tqdm(self.val_loader, desc="Evaluating") as bar:
            for batch in bar:
                loss, acc = self.eval_fn(params, batch)
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
    train_ds = train_ds.map(
        lmrax.datasets.utils.get_map_fn(cfg),
        remove_columns=train_ds.features.keys(),
        num_proc=mp.cpu_count(),
        load_from_cache_file=False,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.model_name)

    val_ds = datasets.load_dataset(cfg.dataset.name, split=cfg.dataset.val)
    ori_val_len = len(val_ds)
    val_ds = val_ds.map(
        lmrax.datasets.utils.get_map_fn(cfg),
        remove_columns=val_ds.features.keys(),
        num_proc=mp.cpu_count(),
        load_from_cache_file=False,
    )
    optimizer_cfg = OmegaConf.to_object(cfg.optimizer)
    optimizer_cls = lmrax.optimizers.get(optimizer_cfg.pop("name"))

    optimizer_chains = [
        optimizer_cls(**optimizer_cfg),
    ]
    if cfg.max_grad_norm is not None:
        optimizer_chains.append(optax.clip_by_global_norm(cfg.max_grad_norm))
    elif cfg.max_grad_value is not None:
        optimizer_chains.append(optax.clip(cfg.max_grad_value))
    optimizer = optax.chain(*optimizer_chains)
    # optimizer = optax.MultiSteps(optimizer, cfg.gradient_accumulation)

    rng = jax.random.PRNGKey(cfg.seed)
    model, params = transformers.FlaxAutoModel.from_pretrained(
        cfg.model_name,
        _do_init=False,
    )
    rng = jax.tree_map(np.asarray, rng)
    params = jax.tree_map(np.asarray, params)
    params = model.init_weights(rng, (1, 1), params)

    trainer = Trainer(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        val_ds=val_ds,
        optimizer=optimizer,
    )

    params, state = trainer.init(params)
    num_params = jax.tree_util.tree_reduce(
        lambda x, y: x + y, jax.tree_map(lambda x: x.size, params)
    )
    wandb.init(project="pf", config=OmegaConf.to_object(cfg), dir=cfg.save_dir)
    wandb.define_metric("val/loss", summary="min")
    wandb.define_metric("val/acc", summary="max")
    wandb.run.config["ori_train_size"] = ori_train_len
    wandb.run.config["ori_val_size"] = ori_val_len
    wandb.run.config["train_size"] = len(train_ds)
    wandb.run.config["val_size"] = len(val_ds)
    wandb.run.config["num_params"] = format(num_params, ",")

    trainer.train(params, state, rng)
    wandb.finish()


if __name__ == "__main__":
    """
    Alternatively, use ```
    jax.config.update("jax_threefry_partitionable", True)
    ``` to reduce the communication overhead.
    """
    jax.config.update("jax_default_prng_impl", "rbg")

    main()
