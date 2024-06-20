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
from ecojax.spaces import Continuous, Discrete


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

    @nn.compact
    def __call__(self, obs: ObservationAgent, key_random: jnp.ndarray) -> ActionAgent:

        # Flatten and concatenate observation inputs
        list_vectors = []
        for name_observation_component, space in self.observation_space_dict.items():
            x: jnp.ndarray = getattr(obs, name_observation_component)
            if isinstance(space, Continuous):
                x = x.reshape((-1,))
                list_vectors.append(x)
            elif isinstance(space, Discrete):
                one_hot_encoded = jax.nn.one_hot(x, space.n)
                list_vectors.append(one_hot_encoded)
            else:
                raise ValueError(f"Unknown space type for observation: {type(space)}")
        x = jnp.concatenate(list_vectors, axis=-1)

        # Process the concatenated output with a final MLP
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(features=hidden_dim)(x)
            x = nn.relu(x)

        # Return the output in the right format
        action, prob = self.get_action_and_prob(x, key_random)
        return action, prob
