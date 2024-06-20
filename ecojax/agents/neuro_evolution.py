from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, List, Tuple, Type

import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from flax import struct
import flax.linen as nn


from ecojax.agents.base_agent_species import BaseAgentSpecies
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
        # If there is no reproduction, return the same state
        if len(dict_reproduction) == 0:
            return batch_state_agents

        # Construct the (filled) lists of indexes of children and parents
        fill_value = self.n_agents_max
        indexes_parents = jnp.full((self.n_agents_max,), fill_value)
        indexes_childrens = jnp.full((self.n_agents_max,), fill_value)
        
        for i, (idx_newborn, list_idx_parents) in enumerate(dict_reproduction.items()):
            if len(list_idx_parents) > 0 and list_idx_parents[0] != -1:
                idx_parent = list_idx_parents[0]
                indexes_childrens = indexes_childrens.at[i].set(idx_newborn)
                indexes_parents = indexes_parents.at[i].set(idx_parent)

        # Apply the reproduction
        key_random, subkey = random.split(key_random)
        batch_state_agents_new = self.manage_reproduction_jitted(
            key_random=subkey,
            batch_state_agents=batch_state_agents,
            indexes_parents=indexes_parents,
            indexes_childrens=indexes_childrens,
        )
        return batch_state_agents_new
    
    
    
    @partial(jax.jit, static_argnums=(0,))
    def manage_reproduction_jitted(
        self,
        key_random: jnp.ndarray,
        batch_state_agents: StateAgent,
        indexes_parents : List[int],
        indexes_childrens : List[int],
    ):
        # Apply the mutation
        batch_keys = random.split(key_random, self.n_agents_max)
        batch_state_agents_mutated = jax.vmap(self.mutate_state_agent)(batch_state_agents, key_random=batch_keys)

        # Transfer the genes from the parents to the childs component by component using jax.tree_map
        def manage_genetic_component_inheritance(genes_target, genes_source):
            return genes_target.at[indexes_childrens].set(genes_source[indexes_parents])

        batch_state_agents_new = jax.tree_map(
            manage_genetic_component_inheritance, batch_state_agents, batch_state_agents_mutated
        )
        return batch_state_agents_new
    
    
    
    def mutate_state_agent(
        self, state_agent: StateAgentEvolutionary, key_random: jnp.ndarray
    ) -> StateAgentEvolutionary:
        return state_agent.replace(
            age=0,
            params=mutation_gaussian_noise(
                arr=state_agent.params,
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
