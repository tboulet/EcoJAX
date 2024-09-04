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
    strength_mutation: float,
    key_random: jnp.ndarray,
) -> jnp.ndarray:
    """Mutates an array by adding Gaussian noise to it.

    Args:
        arr (jnp.ndarray): the array to mutate
        strength_mutation (float): the strength of the mutation, in [0, +oo]
        key_random (jnp.ndarray): the random key used for the mutation

    Returns:
        jnp.ndarray: the mutated array
    """
    return arr + random.normal(key_random, arr.shape) * strength_mutation


def mutate_scalar(
    value: float,
    range: Tuple[Optional[float], Optional[float]],
    key_random: jnp.ndarray,
    mutation_strenth: float = 0.05,
) -> float:
    """Mutates a scalar value.

    Args:
        value (float): the value to mutate
        range (Tuple[Optional[float], Optional[float]]): the range of the value, with None meaning -oo or +oo
        key_random (jnp.ndarray): the random key used for the mutation
        mutation_strenth (float, optional): the strength of the mutation. Defaults to 0.001.
        
    Returns:
        float: the mutated value
    """
    eps = random.normal(key_random) * mutation_strenth
    # Mode [-oo, +oo]:
    if range[0] is None and range[1] is None:
        return value + eps
    # Mode [a, +oo]:
    elif range[0] is not None and range[1] is None:
        a = range[0]
        value_centred = value - a
        value_centred = value_centred * (2 ** eps) 
        return value_centred + a
    # Mode [-oo, b]:
    elif range[0] is None and range[1] is not None:
        b = range[1]
        value_centred = b - value
        value_centred = value_centred * (2 ** eps)
        return b - value_centred
    # Mode [a, b]:
    else:
        a, b = range
        assert a < b, f"Invalid range {range}"
        value_normalized_centered = (value - a) / (b - a)
        value_in_R = logit(value_normalized_centered)
        value_in_R += eps
        value_normalized_centered = sigmoid(value_in_R)
        value = value_normalized_centered * (b - a) + a
        return value
