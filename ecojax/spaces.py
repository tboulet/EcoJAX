from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import random


class EcojaxSpace(ABC):
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
    def get_list_spaces_and_values(
        self, x: Any
    ) -> List[Tuple["EcojaxSpace", jnp.ndarray]]:
        """Flatten the input x to a list of spaces and values."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the dimension of the space."""
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class DiscreteSpace(EcojaxSpace):

    def __init__(self, n: int):
        """A discrete space with n possible values.

        Args:
            n (int): the number of possible values
        """
        assert n > 0, "The number of possible values must be positive."
        assert type(n) == int, "The number of possible values must be an integer."
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

    def get_list_spaces_and_values(self, x: int) -> List[Tuple["EcojaxSpace", int]]:
        return [(self, x)]

    def get_dimension(self) -> int:
        return 1

    def __repr__(self) -> str:
        return f"Discrete({self.n})"


class ContinuousSpace(EcojaxSpace):

    def __init__(
        self,
        shape: Union[int, Tuple[int]],
        low: Optional[float] = None,
        high: Optional[float] = None,
    ):
        """A continuous space with a shape and bounds. For now this is restricted to having all dimensions with the same bounds (while in gym spaces, each dimension can have different bounds).

        Args:
            shape (Union[int, Tuple[int]]): the shape of the space, as a tuple of non-negative integers (or a single integer for 1D spaces)
            low (float): the lower bound of the space
            high (float): the upper bound of the space
        """
        if isinstance(shape, int):
            self.shape = (shape,)
        elif isinstance(shape, tuple):
            self.shape = shape
        else:
            raise ValueError("The shape must be an integer or a tuple of integers.")
        self.low = low
        self.high = high

    def sample(self, key_random: jnp.ndarray) -> float:
        """Sample a value from the space.

        Args:
            key_random (jnp.ndarray): the random key_random

        Returns:
            float: the sampled value
        """
        if self.low is None and self.high is None:
            minval, maxval = -1, 1
        elif self.low is None:
            maxval = self.high
            minval = maxval - 1
        else:
            minval = self.low
            maxval = minval + 1
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
        # Check shape
        if not jnp.shape(x) == self.shape:
            return False
        if self.low != None and jnp.any(x < self.low):
            return False
        if self.high != None and jnp.any(x > self.high):
            return False
        return True

    def get_list_spaces_and_values(self, x: float) -> List[Tuple["EcojaxSpace", float]]:
        return [(self, x)]

    def get_dimension(self) -> int:
        return jnp.prod(jnp.array(self.shape))

    def __repr__(self) -> str:
        minval = self.low if self.low is not None else "-inf"
        maxval = self.high if self.high is not None else "inf"
        return f"Continuous({self.shape} in [{minval}, {maxval}])"


class TupleSpace(EcojaxSpace):

    def __init__(self, tuple_space: Tuple[EcojaxSpace]):
        """A space that is the cartesian product of multiple spaces.

        Args:
            tuple_spaces (Tuple[EcojaxSpace]): the spaces to combine
        """
        self.tuple_spaces = tuple_space

    def sample(self, key_random: jnp.ndarray) -> Tuple[Any]:
        return tuple(space.sample(key_random) for space in self.tuple_spaces)

    def contains(self, x: Tuple[Any]) -> bool:
        if not isinstance(x, tuple):
            return False
        return all(space.contains(x[i]) for i, space in enumerate(self.tuple_spaces))

    def get_list_spaces_and_values(
        self, x: Tuple[Any]
    ) -> List[Tuple["EcojaxSpace", Any]]:
        list_spaces_and_values = []
        for i, space in enumerate(self.tuple_spaces):
            list_spaces_and_values += space.get_list_spaces_and_values(x[i])
        return list_spaces_and_values

    def get_dimension(self) -> int:
        return sum([space.get_dimension() for space in self.tuple_spaces])
    
    def __repr__(self) -> str:
        return f"TupleSpace({', '.join([str(space) for space in self.tuple_spaces])})"


class DictSpace(EcojaxSpace):

    def __init__(self, dict_space: Dict[str, EcojaxSpace]):
        """A space that is the dictionary of multiple spaces.

        Args:
            dict_space (Dict[str, EcojaxSpace]): the spaces to combine
        """
        self.dict_space = dict_space

    def sample(self, key_random: jnp.ndarray) -> Dict[str, Any]:
        return {key: space.sample(key_random) for key, space in self.dict_space.items()}

    def contains(self, x: Dict[str, Any]) -> bool:
        if not isinstance(x, dict):
            return False
        return all(space.contains(x[key]) for key, space in self.dict_space.items())

    def get_list_spaces_and_values(
        self, x: Dict[str, Any]
    ) -> List[Tuple["EcojaxSpace", Any]]:
        list_spaces_and_values = []
        for key, space in self.dict_space.items():
            list_spaces_and_values += space.get_list_spaces_and_values(x[key])
        return list_spaces_and_values

    def get_dimension(self) -> int:
        return sum([space.get_dimension() for space in self.dict_space.values()])
    
    def __repr__(self) -> str:
        return f"DictSpace({', '.join([f'{key}: {str(space)}' for key, space in self.dict_space.items()])})"


class ProbabilitySpace(ContinuousSpace):

    def __init__(self, shape: Union[int, Tuple[int]]):
        """A probability space, i.e. a continuous space with values in [0, 1] and summing to 1."""
        assert (
            isinstance(shape, int) or len(shape) == 1
        ), "The shape of the probability space must be an integer or a tuple of length 1."
        super().__init__(shape, 0, 1)

    def sample(self, key_random: jnp.ndarray) -> jnp.ndarray:
        x = super().sample(key_random)
        return jax.nn.softmax(x)

    def contains(self, x: jnp.ndarray) -> bool:
        return super().contains(x) and jnp.allclose(jnp.sum(x), 1)

    def __repr__(self) -> str:
        return f"ProbabilitySpace({self.shape})"


# if __name__ == "__main__":
#     # Test the spaces
#     key_random = random.PRNGKey(0)
#     discrete = Discrete(5)
#     assert discrete.contains(0)
#     assert discrete.contains(1)
#     assert discrete.contains(2)
#     assert discrete.contains(3)
#     assert discrete.contains(4)
#     assert not discrete.contains(5)
#     assert not discrete.contains(-1)
#     assert discrete.sample(key_random) in [0, 1, 2, 3, 4]

#     continuous = Continuous((), -1, 1)
#     assert continuous.contains(0)
#     assert continuous.contains(0.5)
#     assert continuous.contains(-0.5)
#     assert not continuous.contains(1.5)
#     assert not continuous.contains(-1.5)
#     assert continuous.sample(key_random) >= -1
#     assert continuous.sample(key_random) <= 1

#     tuple_space = TupleSpace(discrete, continuous)
#     assert tuple_space.contains((0, 0))
#     assert tuple_space.contains((4, 1))
#     assert not tuple_space.contains((5, 0))
#     assert not tuple_space.contains((0, 2))
#     assert tuple_space.sample(key_random) == (discrete.sample(key_random), continuous.sample(key_random))

#     dict_space = DictSpace(discrete=discrete, continuous=continuous)
#     assert dict_space.contains({"discrete": 0, "continuous": 0})
#     assert dict_space.contains({"discrete": 4, "continuous": 1})
#     assert not dict_space.contains({"discrete": 5, "continuous": 0})
#     assert not dict_space.contains({"discrete": 0, "continuous": 2})
#     assert dict_space.sample(key_random) == {"discrete": discrete.sample(key_random), "continuous": continuous.sample(key_random)}
