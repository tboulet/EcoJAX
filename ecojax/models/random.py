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

    def obs_to_encoding(
        self, obs: ObservationAgent, key_random: jnp.ndarray
    ) -> jnp.ndarray:
        """Converts the observation to a vector encoding that can be processed by the MLP."""
        return self.space_output.sample(key_random)
