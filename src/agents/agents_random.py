from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, List, Tuple

import jax
from jax import random
import jax.numpy as jnp
import numpy as np

from src.agents import BaseAgentSpecies
from src.types_base import ObservationAgent


class RandomAgentSpecies(BaseAgentSpecies):
    """A species of agents that react randomly to their observations."""

    def __init__(
        self,
        config: Dict,
        n_agents_max: int,
        n_agents_initial: int,
    ):
        super().__init__(
            config=config,
            n_agents_max=n_agents_max,
            n_agents_initial=n_agents_initial,
        )

    def react(
        self,
        key_random: jnp.ndarray,
        batch_observations: ObservationAgent,
        dict_reproduction: Dict[int, List[int]],
    ) -> jnp.ndarray:

        @jax.jit
        def react_single_agent(
            key_random: jnp.ndarray,
            obs: jnp.ndarray,
        ) -> jnp.ndarray:
            return random.randint(key_random, (), 0, 4)

        batch_keys = random.split(key_random, self.n_agents_max)
        react_many_agents = jax.vmap(react_single_agent, in_axes=(0, 0))
        return react_many_agents(
            batch_keys,
            batch_observations,
        )
