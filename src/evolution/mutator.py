from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, List, Tuple, Type

import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from flax import struct
import flax.linen as nn

from src.utils import nest_for_array





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