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

    @nn.compact
    def __call__(self, obs: ObservationAgent, key_random: jnp.ndarray) -> ActionAgent:
        # Flatten and concatenate observation inputs
        flattened_inputs = []
        for name_observation_component, space in self.observation_space_dict.items():
            attribute = getattr(obs, name_observation_component)
            if isinstance(space, Continuous):
                flattened_inputs.append(attribute.flatten())
            elif isinstance(space, Discrete):
                one_hot_encoded = jax.nn.one_hot(attribute, space.n)
                flattened_inputs.append(one_hot_encoded)
            else:
                raise ValueError(f"Unknown space type for observation: {type(space)}")

        concatenated_input = jnp.concatenate(flattened_inputs, axis=-1)

        # Define the hidden layers of the MLP
        for hidden_dim in self.config["hidden_dims"]:
            concatenated_input = nn.Dense(features=hidden_dim)(concatenated_input)
            concatenated_input = nn.relu(concatenated_input)

        # Generate the outputs for each action space
        kwargs_action = {}
        prob_action_sampled = 1.0
        for name_action_component, space in self.action_space_dict.items():
            key_random, subkey = random.split(key_random)
            if isinstance(space, Discrete):
                action_component_logits = nn.Dense(features=space.n)(concatenated_input)
                action_component_probs = nn.softmax(action_component_logits)
                action_component_sampled = jax.random.categorical(
                    subkey, action_component_logits
                )
                kwargs_action[name_action_component] = action_component_sampled
                prob_action_sampled *= action_component_probs[action_component_sampled]
            elif isinstance(space, Continuous):
                mean = nn.Dense(features=np.prod(space.shape))(concatenated_input)
                log_std = nn.Dense(features=np.prod(space.shape))(concatenated_input)
                std = jnp.exp(log_std)
                action_component_sampled = mean + std * random.normal(
                    subkey, shape=mean.shape
                )
                kwargs_action[name_action_component] = action_component_sampled
                # Assuming a standard normal distribution for the purpose of the probability
                action_component_prob = (
                    1.0 / jnp.sqrt(2.0 * jnp.pi * std**2)
                ) * jnp.exp(-0.5 * ((action_component_sampled - mean) / std) ** 2)
                prob_action_sampled *= jnp.prod(action_component_prob)
            else:
                raise ValueError(f"Unknown space type for action: {type(space)}")

        # Return the action
        return self.action_class(**kwargs_action), prob_action_sampled
