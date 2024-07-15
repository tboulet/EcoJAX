from abc import ABC, abstractmethod
from typing import Dict, List, Union
import numpy as np

import jax
from jax import random
import jax.numpy as jnp
from flax import struct

from ecojax.models.base_model import BaseModel
from ecojax.types import ObservationAgent, ActionAgent
from ecojax.spaces import ContinuousSpace, DiscreteSpace


class RandomModel(BaseModel):

    def __call__(self, obs: ObservationAgent, key_random: jnp.ndarray) -> ActionAgent:
        # Generate the outputs for each action space
        kwargs_action = {}
        prob_action_sampled = 1.0

        for name_action_component, space in self.action_space_dict.items():
            key_random, subkey = random.split(key_random)

            if isinstance(space, DiscreteSpace):
                action_component_sampled = random.randint(
                    subkey, shape=(), minval=0, maxval=space.n
                )
                action_component_prob = 1.0 / space.n

            elif isinstance(space, ContinuousSpace):
                action_component_sampled = random.uniform(
                    subkey,
                    shape=space.shape,
                    minval=space.low,
                    maxval=space.high,
                )
                action_component_prob = 1 / (
                    np.prod(space.shape) * (space.high - space.low)
                )
            else:
                raise ValueError(f"Unknown space type for action: {type(space)}")

            kwargs_action[name_action_component] = action_component_sampled
            prob_action_sampled *= action_component_prob

        # Return the action
        return self.action_class(**kwargs_action), prob_action_sampled
