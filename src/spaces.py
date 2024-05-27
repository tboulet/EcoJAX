from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import random


class Space(ABC):
    """The base class for any EcoJAX space. A space describes the valid domain of a variable."""

    @abstractmethod
    def sample(self, key: jnp.ndarray) -> Any:
        """Sample a value from the space.

        Args:
            key (jnp.ndarray): the random key

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
        self.n = n

    def sample(self, key: jnp.ndarray) -> int:
        """Sample a value from the space.

        Args:
            key (jnp.ndarray): the random key

        Returns:
            int: the sampled value
        """
        return random.randint(key, (), 0, self.n)

    def contains(self, x: int) -> bool:
        """Check if a value is in the space.

        Args:
            x (int): the value to check

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

    def sample(self, key: jnp.ndarray) -> float:
        """Sample a value from the space.

        Args:
            key (jnp.ndarray): the random key

        Returns:
            float: the sampled value
        """
        return random.uniform(
            key=key,
            shape=self.shape,
            minval=self.low,
            maxval=self.high,
        )

    def contains(self, x: float) -> bool:
        """Check if a value is in the space.

        Args:
            x (float): the value to check

        Returns:
            bool: whether the value is in the space
        """
        return jnp.logical_and(self.low <= x, x <= self.high)

    def __repr__(self) -> str:
        return f"Continuous({self.shape} in [{self.low}, {self.high}])"
