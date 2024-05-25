from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, List

import jax
from jax import random
import jax.numpy as jnp
import numpy as np


class NeuroEvolutionAgentSpecies:

    def __init__(self, config: Dict):
        self.config = config

    @partial(jax.vmap, in_axes=(None, 0, 0))
    @partial(jax.jit, static_argnums=(0,))
    def single_agent_react(
        self,
        key_random: jnp.ndarray,
        obs: jnp.ndarray,
    ) -> jnp.ndarray:
        """React to a single observation, for a single agent

        Args:
            key_random (jnp.ndarray): the random key, of shape (2,)
            obs (jnp.ndarray): the observation, of shape (**dim_obs)

        Returns:
            jnp.ndarray: the action, of shape (**dim_action)
        """
        return random.randint(key_random, (), 0, 4)

    def react(
        self,
        key_random: jnp.ndarray,
        batch_observations: jnp.ndarray,
    ) -> jnp.ndarray:
        """React to the observations

        Args:
            key_random (jnp.ndarray): the random key, of shape (2,)
            batch_observations (jnp.ndarray): the observations, of shape (n_agents, **dim_obs)

        Returns:
            jnp.ndarray: the actions, of shape (n_agents, **dim_action)
        """
        n_agents = batch_observations.shape[0]
        batch_keys_random = jax.random.split(key_random, n_agents)
        actions = self.single_agent_react(batch_keys_random, batch_observations)
        return actions
