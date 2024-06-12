from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import random


class Space(ABC):
    """The base class for any EcoJAX space. A space describes the valid domain of a variable."""

    @abstractmethod
    def sample(self, key_random: jnp.ndarray) -> Any:
        """Sample a value from the space.

        Args:
            key_random (jnp.ndarray): the random key_random

        Returns:
            Any: the sampled value
        """
        pass

    @abstractmethod
    def contains(self, x: Any) -> bool:
        """Check if a value is in the space.

        Args:
            x (Any): the value to check

        Returns:
            bool: whether the value is in the space
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class Discrete(Space):

    def __init__(self, n: int):
        """The constructor of the Discrete class.

        Args:
            n (int): the number of possible values
        """
        assert n > 0, "The number of possible values must be positive."
        self.n = n

    def sample(self, key_random: jnp.ndarray) -> int:
        """Sample a value from the space.

        Args:
            key_random (jnp.ndarray): the random key_random

        Returns:
            int: the sampled value
        """
        return random.randint(key_random, (), 0, self.n)

    def contains(self, x: jax.Array) -> bool:
        """Check if a value is in the space.

        Args:
            x (jax.Array): the value to check

        Returns:
            bool: whether the value is in the space
        """
        return jnp.logical_and(0 <= x < self.n, jnp.equal(x, jnp.floor(x)))

    def __repr__(self) -> str:
        return f"Discrete({self.n})"


class Continuous(Space):

    def __init__(
        self,
        shape: Union[int, Tuple[int]],
        low: float,
        high: float,
    ):
        """The constructor of the Continuous class.

        Args:
            shape (Union[int, Tuple[int]]): the shape of the space, or the number of dimensions
            low (float): the lower bound of the space
            high (float): the upper bound of the space
        """
        self.shape = shape
        self.low = low
        self.high = high

    def sample(self, key_random: jnp.ndarray) -> float:
        """Sample a value from the space.

        Args:
            key_random (jnp.ndarray): the random key_random

        Returns:
            float: the sampled value
        """
        minval=self.low if self.low is not None else -1
        maxval=self.high if self.high is not None else 1
        return random.uniform(
            key=key_random,
            shape=self.shape,
            minval=minval,
            maxval=maxval,
        )

    def contains(self, x: float) -> bool:
        """Check if a value is in the space.

        Args:
            x (float): the value to check

        Returns:
            bool: whether the value is in the space
        """
        # CHeck shape
        if not jnp.shape(x) == self.shape:
            return False
        if self.low != None and jnp.any(x < self.low):
            return False
        if self.high != None and jnp.any(x > self.high):
            return False
        return True
    
    def __repr__(self) -> str:
        minval=self.low if self.low is not None else "-inf"
        maxval=self.high if self.high is not None else "inf"
        return f"Continuous({self.shape} in [{minval}, {maxval}])"
