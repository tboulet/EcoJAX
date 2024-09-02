from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Type

import jax
import numpy as np

import flax.linen as nn
from jax import random
import jax.numpy as jnp

from ecojax.spaces import (
    ContinuousSpace,
    DictSpace,
    DiscreteSpace,
    EcojaxSpace,
    ProbabilitySpace,
    TupleSpace,
)
from ecojax.types import ActionAgent, ObservationAgent
from ecojax.utils import jprint

name_activation_fn_to_fn = {
    "relu": nn.relu,
    "sigmoid": nn.sigmoid,
    "tanh": nn.tanh,
    "leaky_relu": nn.leaky_relu,
    "elu": nn.elu,
    "selu": nn.selu,
    "gelu": nn.gelu,
    "swish": nn.swish,
    "identity": lambda x: x,
    "linear": lambda x: x,
    None: lambda x: x,
}


class BaseModel(nn.Module, ABC):
    """The base class for all models. A model is a way to map observations to actions.
    This abstract class subclasses nn.Module, which is the base class for all Flax models.

    For subclassing this class, users need to add the dataclass parameters and implement the obs_to_encoding method.

    Args:
        space_input (EcojaxSpace): the input space of the model
        space_output (EcojaxSpace): the output space of the model
    """

    space_input: EcojaxSpace
    space_output: EcojaxSpace

    @abstractmethod
    def obs_to_encoding(
        self, obs: ObservationAgent, key_random: jnp.ndarray
    ) -> jnp.ndarray:
        """Converts the observation to a vector encoding that can be processed by the model."""
        pass

    def get_initialized_variables(
        self, key_random: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        """Initializes the model's variables and returns them as a dictionary.
        This is a wrapper around the init method of nn.Module, which creates an observation for initializing the model.
        """
        # Sample the observation from the different spaces
        key_random, subkey = random.split(key_random)
        x = self.space_input.sample(subkey)

        # Run the forward pass to initialize the model
        key_random, subkey = random.split(key_random)
        return nn.Module.init(
            self,
            key_random,
            x=x,
            key_random=subkey,
        )

    def process_encoding(self, x: jnp.ndarray, key_random: jnp.ndarray) -> jnp.ndarray:
        """Processes the encoding to obtain the output of the model."""
        assert isinstance(
            x, jnp.ndarray
        ), f"Encoding must be a jnp.ndarray, not {type(x)}"
        assert len(x.shape) == 1, f"Encoding must be a 1D array, not {x.shape}"
        # Process the encoding to obtain the output
        if isinstance(self.space_output, DiscreteSpace):
            logits = nn.Dense(
                features=self.space_output.n,
            )(x)
            output = random.categorical(
                key_random, logits
            )  # crashes here because non supported operation
        elif isinstance(self.space_output, ContinuousSpace):
            shape_output = self.space_output.shape
            d_encoding = x.shape[0]
            # (d_encoding,) -> ?
            if len(shape_output) == 1:
                d_output = shape_output[0]
                # (d_encoding,) -> (n,)
                if d_output == d_encoding:
                    # (d_encoding,) -> (d_encoding,) : just return the value
                    output = x
                else:
                    # (d_encoding,) -> (d_output,) : apply dense layer
                    output = nn.Dense(
                        features=d_output,
                        # bias_init=nn.initializers.ones_init(), # TODO : optionalize this, and find a way so that zeros doesnt create NaNs
                        bias_init=nn.initializers.zeros_init(),
                    )(x)
            elif len(shape_output) == 0:
                # (d_encoding,) -> ()
                if len(x.shape) == 1:
                    # (1,) -> () : just extract the value
                    output = x[0]
                else:
                    # (d_encoding,) -> () : apply dense layer and extract the value
                    output = nn.Dense(
                        features=1,
                        bias_init=nn.initializers.ones_init(),
                    )(x)[0]
            else:
                raise NotImplementedError(
                    f"Processing of continuous space of shape {shape_output} is not implemented."
                )
        elif isinstance(self.space_output, TupleSpace):
            subkeys = random.split(key_random, len(self.space_output.tuple_spaces))
            return tuple(
                self.process_encoding(x, subkeys[i])
                for i in range(len(self.space_output.tuple_spaces))
            )
        elif isinstance(self.space_output, DictSpace):
            subkeys = random.split(key_random, len(self.space_output.dict_space))
            return {
                key: self.process_encoding(x, subkeys[i])
                for i, key in enumerate(self.space_output.dict_space.keys())
            }
        else:
            raise ValueError(
                f"Unknown space type for output: {type(self.space_output)}"
            )

        # Return the output
        assert self.space_output.contains(
            output
        ), f"Output {output} is not in the output space {self.space_output}"
        return output

    def activation_fn(self, name_activation_fn, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the activation function to the input."""
        return name_activation_fn_to_fn[name_activation_fn](x)

    @nn.compact
    def __call__(
        self,
        x: Any,
        key_random: jnp.ndarray,
    ) -> Tuple[jnp.ndarray]:
        """The forward pass of the model. It maps the observation to the output in the right format.

        Args:
            x (Any) : input observation
            key_random (jnp.ndarray): the random key used for any random operation in the forward pass

        Returns:
            Tuple[jnp.ndarray]: a tuple of the requested outputs
        """
        # Convert the observation to a vector encoding
        encoding = self.obs_to_encoding(x, key_random)

        # Return the output in the desired output space
        output = self.process_encoding(encoding, key_random)
        return output

    def get_table_summary(self) -> Dict[str, Any]:
        """Returns a table that summarizes the model's parameters and shapes."""
        key_random = jax.random.PRNGKey(0)
        x = self.space_input.sample(key_random)
        return nn.tabulate(self, rngs=key_random)(x, key_random)
