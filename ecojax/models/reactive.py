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
from ecojax.utils import sigmoid


class ReactiveModel(BaseModel):
    """A class of model that react in a scripted, hard-coded way to the observations."""

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
        
        # Get the greedy action
        condlist = []
        choicelist = []
        for name_action in self.list_actions:
            if name_action == "right":
                condlist.append(visual_field_plants[v, v+1] == 1)
                choicelist.append(self.list_actions.index(name_action))
            elif name_action == "left":
                condlist.append(visual_field_plants[v, v-1] == 1)
                choicelist.append(self.list_actions.index(name_action))
            elif name_action == "backward":
                condlist.append(visual_field_plants[v-1, v] == 1)
                choicelist.append(self.list_actions.index(name_action))
        idx_action = jnp.select(
            condlist=condlist,
            choicelist=choicelist,
            default=self.list_actions.index("forward")
        )
        
        # Play epsilon-greedy strategy
        factor_randomness = self.param(
            name="factor_randomness",
            init_fn=lambda _ : jnp.array(0),
        )
        prob = sigmoid(factor_randomness)
        key_random, subkey = jax.random.split(key_random)
        idx_action = jax.lax.cond(
            jax.random.uniform(subkey) < prob,
            lambda _: jax.random.randint(subkey, (), 0, len(self.list_actions)),
            lambda _: idx_action,
            operand=None,
        )
        
        # Return a one-hot encoding of the action
        logits = jnp.full((n_actions,), -np.inf)
        logits = logits.at[idx_action].set(np.inf)
        return logits
