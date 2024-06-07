from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, List, Tuple, Type

import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from flax import struct
import flax.linen as nn


from ecojax.agents.base_agent_species import BaseAgentSpecies, set_state
from ecojax.models.base_model import BaseModel
from ecojax.evolution.mutator import mutation_gaussian_noise
from ecojax.types import ActionAgent, ObservationAgent, StateAgent
import ecojax.spaces as spaces


@struct.dataclass
class StateAgentEvolutionary(StateAgent):
    # The age of the agent, in number of timesteps
    age: int

    # The parameters of the neural network corresponding to the agent
    params: Dict[str, jnp.ndarray]


class NeuroEvolutionAgentSpecies(BaseAgentSpecies):
    """A species of agents that evolve their neural network weights."""

    def init(self, key_random: jnp.ndarray) -> None:

        # Initialize the state
        def init_single_agent(
            model: BaseModel,
            key_random: jnp.ndarray,
        ) -> jnp.ndarray:
            variables = model.get_initialized_variables(key_random)
            params = variables.get("params", {})
            return StateAgentEvolutionary(age=0, params=params)

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
        key_random, subkey = random.split(key_random)
        self.batch_state_agents = self.manage_reproduction(
            key_random=subkey,
            batch_state_agents=self.batch_state_agents,
            dict_reproduction=dict_reproduction,
        )

        # Agent-wise reaction
        key_random, subkey = random.split(key_random)
        self.batch_state_agents, batch_actions = self.react_agents(
            key_random=subkey,
            batch_observations=batch_observations,
            batch_state_agents=self.batch_state_agents,
        )

        return batch_actions

    # =============== Reproduction methods =================

    def manage_reproduction(
        self,
        key_random: jnp.ndarray,
        batch_state_agents: StateAgent,
        dict_reproduction: Dict[int, List[int]],
    ) -> StateAgent:
        """Manage the reproduction of the agents.

        Args:
            key_random (jnp.ndarray): the random key, of shape (2,)
            batch_state_agents (StateAgent): the state of the agents
            dict_reproduction (Dict[int, List[int]]): the dictionary indicating the indexes of the parents of each newborn agent

        Returns:
            StateAgent: the updated state of the agents, with the newborn agents added
        """
        for idx_newborn, list_idx_parents in dict_reproduction.items():
            # Get the parent's AgentState
            idx_parent = list_idx_parents[0]
            state_parent = jax.tree_map(lambda x: x[idx_parent], batch_state_agents)
            # Mutate the parent's AgentState to create the newborn
            key_random, subkey = random.split(key_random)
            state_mutated = self.mutate_agent(state_parent, key_random=subkey)
            # Add the newborn to the batch
            batch_state_agents = jax.tree_map(
                lambda x, y: x.at[idx_newborn].set(y), batch_state_agents, state_mutated
            )
        return batch_state_agents

    def mutate_agent(
        self, agent: StateAgentEvolutionary, key_random: jnp.ndarray
    ) -> StateAgentEvolutionary:
        return agent.replace(
            age=0,
            params=mutation_gaussian_noise(
                arr=agent.params,
                mutation_rate=0.1,
                mutation_std=0.01,
                key_random=key_random,
            ),
        )

    # =============== Agent creation methods =================

    @partial(jax.jit, static_argnums=(0,))
    def react_agents(
        self,
        key_random: jnp.ndarray,
        batch_observations: ObservationAgent,  # Batched
        batch_state_agents: StateAgentEvolutionary,  # Batched
    ) -> jnp.ndarray:

        def react_single_agent(
            key_random: jnp.ndarray,
            obs: jnp.ndarray,
            state_agent: StateAgentEvolutionary,
        ) -> jnp.ndarray:
            # Inference part
            action, prob_action = self.model.apply(
                variables={"params": state_agent.params},
                obs=obs,
                key_random=key_random,
            )
            # Learning part
            state_agent.replace(age=state_agent.age + 1)
            # Update the agent's state and act
            return state_agent, action

        react_many_agents = jax.vmap(react_single_agent, in_axes=(0, 0, 0))
        batch_keys = random.split(key_random, self.n_agents_max)
        batch_state_agents, batch_actions = react_many_agents(
            batch_keys,
            batch_observations,
            batch_state_agents,
        )

        return batch_state_agents, batch_actions
