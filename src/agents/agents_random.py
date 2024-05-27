from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, List, Tuple, Type

import jax
from jax import random
import jax.numpy as jnp
import numpy as np

from src.agents import BaseAgentSpecies
from src.types_base import ActionAgent, ObservationAgent
import src.spaces as spaces


class RandomAgentSpecies(BaseAgentSpecies):
    """A species of agents that react randomly to their observations."""

    def __init__(
        self,
        config: Dict,
        n_agents_max: int,
        n_agents_initial: int,
        observation_space_dict: Dict,
        action_space_dict: Dict,
        observation_class: Type[ObservationAgent],
        action_class: Type[ActionAgent],
        
    ):
        super().__init__(
            config=config,
            n_agents_max=n_agents_max,
            n_agents_initial=n_agents_initial,
            observation_space_dict=observation_space_dict,
            action_space_dict=action_space_dict,
            observation_class=observation_class,
            action_class=action_class,
        )
        
    @partial(jax.jit, static_argnums=(0,))
    @partial(jax.vmap, in_axes=(None, 0, 0))
    def react_to_obs_agent(
        self,
        key_random: jnp.ndarray,
        obs: jnp.ndarray,
    ) -> jnp.ndarray:

        kwargs_action = {}
        subkeys = random.split(key_random, len(self.action_space_dict))
        for idx, (component_action, space) in enumerate(
            self.action_space_dict.items()
        ):
            if type(space) == spaces.Discrete:
                kwargs_action[component_action] = space.sample(
                    key=subkeys[idx]
                )
            elif type(space) == spaces.Continuous:
                kwargs_action[component_action] = space.sample(
                    key=subkeys[idx]
                )
            else:
                raise ValueError(
                    f"Space type {type(space)} not supported."
                )
        return self.action_class(**kwargs_action)
            

    def react(
        self,
        key_random: jnp.ndarray,
        batch_observations: ObservationAgent,
        dict_reproduction: Dict[int, List[int]],
    ) -> jnp.ndarray:
        
        # Generate a batch of random keys
        batch_keys = random.split(key_random, self.n_agents_max)
        # React to the observations
        direction = self.react_to_obs_agent(
            batch_keys,
            batch_observations,
        )
        return direction