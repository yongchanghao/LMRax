"""
Wrappers for optax optimizers to make them compatible with LMRax.
Added Lion.
"""

from optax import (
    adabelief,
    adafactor,
    adagrad,
    adam,
    adamax,
    adamaxw,
    adamw,
    constant_schedule,
    lamb,
    lars,
    novograd,
    radam,
    rmsprop,
    sgd,
    warmup_cosine_decay_schedule,
)

from lmrax.optimizers.lion import lion

__OPTIMIZERS__ = {
    "adam": adam,
    "adamw": adamw,
    "adabelief": adabelief,
    "adafactor": adafactor,
    "adagrad": adagrad,
    "adamax": adamax,
    "adamaxw": adamaxw,
    "lamb": lamb,
    "lars": lars,
    "lion": lion,
    "novograd": novograd,
    "radam": radam,
    "rmsprop": rmsprop,
    "sgd": sgd,
}

__SCHEDULERS__ = {
    "constant": constant_schedule,
    "warmup_cosine_decay": warmup_cosine_decay_schedule,
}


def get_optimizer(name):
    return __OPTIMIZERS__[name]


def get_scheduler(name):
    return __SCHEDULERS__[name]
