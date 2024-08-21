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
from ecojax.utils import jprint, sigmoid


class HumanModel(BaseModel):
    """A model that gives control to the human player through CLI."""

    list_actions: List[str]
    list_channels_visual_field: List[str]
    
    def obs_to_encoding(
        self, obs: ObservationAgent, key_random: jnp.ndarray
    ) -> jnp.ndarray:
        """Directly convert the observation to the action, which will be transmitted directly by the model."""
        assert isinstance(
            self.space_output, ContinuousSpace
        ), "The output space must be a continuous space."
        assert (
            len(self.space_output.shape) == 1
        ), "The output space must be a 1D continuous space of shape (n_actions,)."
                
        assert "forward" in self.list_actions, "The action 'forward' must be in the list of actions."
        assert "plants" in self.list_channels_visual_field, "The channel 'plants' must be in the list of visual field channels."
        n_actions = len(self.list_actions)
        idx_plant = self.list_channels_visual_field.index("plants")
        visual_field_plants = obs["visual_field"][:, :, idx_plant]
        v = visual_field_plants.shape[0] // 2
        
        # Print the visual field
        print(visual_field_plants)
        print(obs)
        jprint(visual_field_plants)
        jprint(obs)
        
        # Get the greedy action
        idx_action = input()
        idx_action = int(idx_action)
        
        # Return a one-hot encoding of the action
        logits = jnp.full((n_actions,), -np.inf)
        logits = logits.at[idx_action].set(np.inf)
        return logits
