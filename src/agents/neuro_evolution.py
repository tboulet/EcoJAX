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

    @partial(jax.jit, static_argnums=(0,))
    @partial(jax.vmap, in_axes=(None, 0, 0))
    def single_agent_react(
        self,
        key_random : jnp.ndarray,
        observation : jnp.ndarray,
    ) -> jnp.ndarray:
        """React to a single observation, for a single agent

        Args:
            key_random (jnp.ndarray): the random key, of shape (2,)
            observation (jnp.ndarray): the observation, of shape (**dim_observation)
            
        Returns:
            jnp.ndarray: the action, of shape (**dim_action)
        """

        return random.normal(key_random, shape=())

    
    def react(
        self,
        key_random: jnp.ndarray,
        observations: jnp.ndarray,
    ) -> jnp.ndarray:
        """React to the observations

        Args:
            key_random (jnp.ndarray): the random key
            observations (jnp.ndarray): the observations, of shape (**dim_observation)

        Returns:
            jnp.ndarray: the actions, of shape (**dim_action)
        """
        batch_size = observations.shape[0]
        keys = jax.random.split(key_random, batch_size)
        actions = self.single_agent_react(keys, observations)
        return actions
