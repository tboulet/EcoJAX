from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, List, Tuple, Type

import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from flax import struct
import flax.linen as nn


from src.agents import BaseAgentSpecies
from src.models.base_model import BaseModel
from src.types_base import ActionAgent, ObservationAgent, StateAgent
import src.spaces as spaces


@struct.dataclass
class StateAgentEvolutionary(StateAgent):
    # The age of the agent, in number of timesteps
    age: int

    # The weights of the neural network corresponding to the agent
    weights: jnp.ndarray


class NeuroEvolutionAgentSpecies(BaseAgentSpecies):
    """A species of agents that evolve their neural network weights."""

    def start(self, key_random: jnp.ndarray) -> None:
        pass

        # Initialize the state
        def init_single_agent(
            model: BaseModel,
            key_random: jnp.ndarray,
        ) -> jnp.ndarray:
            variables = model.get_initialized_variables(key_random)
            return StateAgentEvolutionary(age=0, weights=variables["params"])

        key_random, subkey = random.split(key_random)
        batch_keys = jax.random.split(subkey, self.n_agents_max)
        init_many_agents = jax.vmap(init_single_agent, in_axes=(None, 0))
        self.batch_state_agents = init_many_agents(
            self.model,
            batch_keys,
        )

    def react(
        self,
        key_random: jnp.ndarray,
        batch_observations: ObservationAgent,  # Batched
        dict_reproduction: Dict[int, List[int]],
    ) -> jnp.ndarray:

        # Reproduction part
        # for idx_agent_newborn, list_idx_parents in dict_reproduction.items():
        #     if len(list_idx_parents) == 0:
        #         agent = self.create_random_agent(key_random)
        #     elif len(list_idx_parents) == 1:
        #         agent = self.create_mutated_agent(key_random, list_idx_parents[0])
        #     elif len(list_idx_parents) == 2:
        #         agent = self.create_crossover_agent(
        #             key_random, list_idx_parents[0], list_idx_parents[1]
        #         )
        #     else:
        #         raise ValueError(
        #             f"list_idx_parents has length {len(list_idx_parents)}. Not supported."
        #         )

        # Agent-wise acting part
        batch_keys = random.split(key_random, self.n_agents_max)
        self.batch_state_agents, batch_actions = self.react_agents(
            batch_keys=batch_keys,
            batch_observations=batch_observations,
            batch_state_agents=self.batch_state_agents,
        )

        return batch_actions

    # =============== Reproduction methods =================

    # =============== Agent creation methods =================

    @partial(jax.jit, static_argnums=(0,))
    def react_agents(
        self,
        batch_keys: jnp.ndarray,
        batch_observations: ObservationAgent,  # Batched
        batch_state_agents: StateAgentEvolutionary,  # Batched
    ) -> jnp.ndarray:

        def react_single_agent(
            key_random: jnp.ndarray,
            obs: jnp.ndarray,
            state_agent: StateAgentEvolutionary,
        ) -> jnp.ndarray:
            action, prob_action = self.model.apply(
                variables={"params": state_agent.weights},
                obs=obs,
                key_random=key_random,
            )
            return state_agent, action

        react_many_agents = jax.vmap(react_single_agent, in_axes=(0, 0, 0))
        batch_state_agents, batch_actions = react_many_agents(
            batch_keys,
            batch_observations,
            batch_state_agents,
        )
        
        return batch_state_agents, batch_actions
