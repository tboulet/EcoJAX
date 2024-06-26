from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Type

import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from flax import struct
import flax.linen as nn

from ecojax.utils import logit, nest_for_array, sigmoid


@nest_for_array
def mutation_gaussian_noise(
    arr: jnp.ndarray,
    mutation_rate: float,
    mutation_std: float,
    key_random: jnp.ndarray,
) -> jnp.ndarray:
    """Mutates an array by adding Gaussian noise to it.

    Args:
        arr (jnp.ndarray): the array to mutate
        mutation_rate (float): the probability of mutating each element of the array
        mutation_std (float): the standard deviation of the Gaussian noise
        key_random (jnp.ndarray): the random key used for the mutation

    Returns:
        jnp.ndarray: the mutated array
    """
    key_random, subkey = random.split(key_random)
    mask_mutation = random.bernoulli(subkey, mutation_rate, shape=arr.shape)
    mutation = random.normal(subkey, arr.shape) * mutation_std
    return arr + mask_mutation * mutation


def mutate_scalar(
    value: float,
    range: Tuple[Optional[float], Optional[float]],
    key_random: jnp.ndarray,
) -> float:
    """Mutates a scalar value.

    Args:
        value (float): the value to mutate
        range (Tuple[Optional[float], Optional[float]]): the range of the value, with None meaning -oo or +oo
        key_random (jnp.ndarray): the random key used for the mutation

    Returns:
        float: the mutated value
    """
    # Mode [-oo, +oo]:
    if range[0] is None and range[1] is None:
        return value + random.normal(key_random)
    # Mode [a, +oo]:
    elif range[0] is not None and range[1] is None:
        a = range[0]
        value_centred = value - a
        value_centred *= jnp.exp(random.normal(key_random))
        return value_centred + a
    # Mode [-oo, b]:
    elif range[0] is None and range[1] is not None:
        b = range[1]
        value_centred = b - value
        value_centred *= jnp.exp(random.normal(key_random))
        return b - value_centred
    # Mode [a, b]:
    else:
        a, b = range
        assert a < b, f"Invalid range {range}"
        value_normalized_centered = (value - a) / (b - a)
        value_in_R = logit(value_normalized_centered)
        value_in_R += random.normal(key_random)
        value_normalized_centered = sigmoid(value_in_R)
        value = value_normalized_centered * (b - a) + a
        return value
