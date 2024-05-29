from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np

import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from flax import struct
import flax.linen as nn

from src.models.base_model import BaseModel
from src.types_base import ObservationAgent, ActionAgent


class MLP_Model(BaseModel):

    @nn.compact
    def __call__(self, obs: ObservationAgent, key_random: jnp.ndarray) -> ActionAgent:

        # Flatten and concatenate observation inputs
        flattened_inputs = []
        for name_action_component, space in self.observation_space_dict.items():
            attribute = getattr(obs, name_action_component)
            flattened_inputs.append(attribute.flatten())

        concatenated_input = jnp.concatenate(flattened_inputs, axis=-1)

        # Define the hidden layers of the MLP
        for hidden_dim in self.config["hidden_dims"]:
            concatenated_input = nn.Dense(features=hidden_dim)(concatenated_input)
            concatenated_input = nn.relu(concatenated_input)

        # Generate the outputs for each action space
        kwargs_action = {}
        prob_action_sampled = 1.0
        for name_action_component, space in self.action_space_dict.items():
            # Generate the logits for the action space
            action_component_logits = nn.Dense(features=space.n)(concatenated_input)
            action_component_probs = nn.softmax(action_component_logits)
            # Sample an action
            action_component_sampled = jax.random.categorical(
                key_random, action_component_probs
            )
            kwargs_action[name_action_component] = action_component_sampled
            # Also compute the probability of the sampled action
            prob_action_sampled *= action_component_probs[action_component_sampled]

        # Return the action
        return self.action_class(**kwargs_action), prob_action_sampled
