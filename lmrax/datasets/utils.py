import random
import math

import numpy as np
import jax
import datasets


def seed_worker(seed):
    random.seed(seed)
    np.random.seed(seed)


def shp_to_pf(example):
    context = example["history"]
    chosen = example["human_ref_A"] if example["labels"] == 1 else example["human_ref_B"]
    rejected = example["human_ref_B"] if example["labels"] == 1 else example["human_ref_A"]
    s = example["score_ratio"]
    s = s / (s + 1)
    weight = s
    return {
        "context": context,
        "chosen": chosen,
        "rejected": rejected,
        "weight": weight,
    }


def hh_to_pf(examples):
    def per_example_fn(chosen, rejected):
        chosen_turns = [x.strip() for x in chosen.split("\n\nAssistant") if len(x.strip()) > 0]
        rejected_turns = [x.strip() for x in rejected.split("\n\nAssistant") if len(x.strip()) > 0]

        context_idx = 0
        for i in range(min(len(chosen_turns), len(rejected_turns))):
            if chosen_turns[i] == rejected_turns[i]:
                context_idx = i + 1
            else:
                break
        if not 0 < context_idx < min(len(chosen_turns), len(rejected_turns)):
            assert chosen == rejected
            context_idx -= 1

        context = "\n\nAssistant".join(chosen_turns[:context_idx]).strip()
        chosen = "\n\nAssistant".join([""] + chosen_turns[context_idx:]).strip()
        rejected = "\n\nAssistant".join([""] + rejected_turns[context_idx:]).strip()

        weight = 1.0

        return {
            "context": context,
            "chosen": chosen,
            "rejected": rejected,
            "weight": weight,
        }

    results = [per_example_fn(chosen, rejected) for chosen, rejected in zip(examples["chosen"], examples["rejected"])]

    return {
        "context": [x["context"] for x in results],
        "chosen": [x["chosen"] for x in results],
        "rejected": [x["rejected"] for x in results],
        "weight": [x["weight"] for x in results],
    }


def get_filter_fn(cfg):
    name = cfg.dataset.name
    if name == "stanfordnlp/SHP":
        return lambda x: x["score_ratio"] >= cfg.dataset.score_ratio
    elif name == "Anthropic/hh-rlhf":
        return lambda x: x["chosen"] != x["rejected"]
    else:
        return lambda x: True


def get_map_fn(cfg):
    name = cfg.dataset.name
    if name == "stanfordnlp/SHP":
        return shp_to_pf
    elif name == "Anthropic/hh-rlhf":
        return hh_to_pf
    else:
        raise NotImplementedError(
            f"Dataset {name} is not supported yet. "
            f"Please open an issue. "
            f"You can also implement it and send a PR."
        )


def preprocess_function(examples, cfg, tokenizer, max_source_length, max_target_length):
    examples = get_map_fn(cfg)(examples)
    context = examples["context"]
    chosen = examples["chosen"]
    rejected = examples["rejected"]
    weight = examples["weight"]

    context_ids = tokenizer(
        context,
        max_length=max_source_length,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )

    chosen_ids = tokenizer(
        text_target=chosen,
        max_length=max_target_length,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )

    rejected_ids = tokenizer(
        text_target=rejected,
        max_length=max_target_length,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )

    return {
        "context_input_ids": context_ids["input_ids"],
        "context_attention_mask": context_ids["attention_mask"],
        "chosen_input_ids": chosen_ids["input_ids"],
        "chosen_attention_mask": chosen_ids["attention_mask"],
        "rejected_input_ids": rejected_ids["input_ids"],
        "rejected_attention_mask": rejected_ids["attention_mask"],
        "weight": np.asarray(weight),
    }


def data_loader(
    rng: jax.random.PRNGKey, dataset: datasets.Dataset, batch_size: int, shuffle: bool = False, drop_last=True
):
    if shuffle:
        batch_idx = jax.random.permutation(rng, len(dataset))
        batch_idx = np.asarray(batch_idx)
    else:
        batch_idx = np.arange(len(dataset))

    if drop_last:
        steps_per_epoch = len(dataset) // batch_size
        batch_idx = batch_idx[: steps_per_epoch * batch_size]  # Skip incomplete batch.
        batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))
    else:
        steps_per_epoch = math.ceil(len(dataset) / batch_size)
        batch_idx = np.array_split(batch_idx, steps_per_epoch)

    for idx in batch_idx:
        batch = dataset[idx]
        batch = {k: np.array(v) for k, v in batch.items()}

        yield batch
