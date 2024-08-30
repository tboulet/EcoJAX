from abc import ABC, abstractmethod
from typing import Dict, List, Union
import numpy as np

import jax
from jax import random
import jax.numpy as jnp
from flax import struct
import flax.linen as nn

from ecojax.models.base_model import BaseModel
from ecojax.types import ObservationAgent, ActionAgent
from ecojax.spaces import ContinuousSpace, DiscreteSpace
from ecojax.utils import jprint


class MLP_Model(BaseModel):
    """A model that use MLP networks to process observations and generate actions.
    It does the following :
    - flatten and concatenate the observation components to obtain a single vector
    - process the concatenated output with a final MLP to obtain an output of shape (hidden_dims[-1],)
    - generate the outputs for each action space through finals MLPs

    Args:
        hidden_dims (List[int]): the number of hidden units in each hidden layer. It also defines the number of hidden layers.
    """

    hidden_dims: List[int]
    name_activation_fn: str = "relu"

    def obs_to_encoding(
        self, obs: ObservationAgent, key_random: jnp.ndarray
    ) -> jnp.ndarray:
        """Converts the observation to a vector encoding that can be processed by the MLP."""

        # Flatten and concatenate observation inputs
        list_spaces_and_values = self.space_input.get_list_spaces_and_values(obs)
        list_vectors = []
        for space, x in list_spaces_and_values:
            if isinstance(space, ContinuousSpace):
                x = x.reshape((-1,))
                list_vectors.append(x)
            elif isinstance(space, DiscreteSpace):
                one_hot_encoded = jax.nn.one_hot(x, space.n)
                list_vectors.append(one_hot_encoded)
            else:
                raise ValueError(f"Unknown space type for observation: {type(space)}")
        x = jnp.concatenate(list_vectors, axis=-1)

        # Process the concatenated output with a final MLP
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(
                features=hidden_dim,
            )(x)
            x = self.activation_fn(name_activation_fn=self.name_activation_fn, x=x)

        print(f"obs: {obs}")
        return x
