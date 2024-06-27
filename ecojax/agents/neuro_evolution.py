from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, List, Tuple, Type

import jax
from jax import random, tree_map
import jax.numpy as jnp
import numpy as np
from flax import struct
import flax.linen as nn


from ecojax.agents.base_agent_species import BaseAgentSpecies
from ecojax.core.eco_info import EcoInformation
from ecojax.models.base_model import BaseModel
from ecojax.evolution.mutator import mutation_gaussian_noise
from ecojax.types import ActionAgent, ObservationAgent
import ecojax.spaces as spaces
from ecojax.utils import jprint


@struct.dataclass
class StateAgentEvolutionary:
    # The age of the agent, in number of timesteps
    age: int

    # The parameters of the neural network corresponding to the agent
    params: Dict[str, jnp.ndarray]


class NeuroEvolutionAgentSpecies(BaseAgentSpecies):
    """A species of agents that evolve their neural network weights."""

    def reset(self, key_random: jnp.ndarray) -> None:

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
        eco_information: EcoInformation,
    ) -> ActionAgent:

        self.batch_state_agents, batch_actions = self.react_jitted(
            key_random=key_random,
            batch_state_agents=self.batch_state_agents,
            batch_observations=batch_observations,
            eco_information=eco_information,
        )

        return batch_actions

    @partial(jax.jit, static_argnums=(0,))
    def react_jitted(
        self,
        key_random: jnp.ndarray,
        batch_state_agents: StateAgentEvolutionary,  # Batched
        batch_observations: ObservationAgent,  # Batched
        eco_information: EcoInformation,
    ) -> jnp.ndarray:

        # Apply the mutation
        batch_keys = random.split(key_random, self.n_agents_max)
        batch_state_agents_mutated = jax.vmap(self.mutate_state_agent)(
            batch_state_agents, key_random=batch_keys
        )

        # Transfer the genes from the parents to the childs component by component using jax.tree_map
        are_newborns_agents = eco_information.are_newborns_agents
        indexes_parents = eco_information.indexes_parents  # (n_agents, n_parents)
        if indexes_parents.shape == (self.n_agents_max, 1):
            indexes_parents = indexes_parents.squeeze(axis=-1)  # (n_agents,)
        else:
            raise NotImplementedError(
                f"Invalid shape for indexes_parents: {indexes_parents.shape}"
            )

        def manage_genetic_component_inheritance(
            genes: jnp.ndarray,
            genes_mutated: jnp.ndarray,
        ):
            """The function that manages the inheritance of genetic components.
            It will be apply to each genetic component (JAX array of shape (n_agents, *gene_shape)) of the agents.
            It replace the genes by the mutated genes of the parents but only for the newborn agents.

            Args:
                genes (jnp.ndarray): a JAX array of shape (n_agents, *gene_shape)
                genes_mutated (jnp.ndarray): a JAX array of shape (n_agents, *gene_shape), which are mutated genes

            Returns:
                genes_inherited (jnp.ndarray): a JAX array of shape (n_agents, *gene_shape), which are the genes of the population after inheritance
            """
            mask = are_newborns_agents
            for _ in range(genes.ndim - 1):
                mask = jnp.expand_dims(mask, axis=-1)
            genes_inherited = jnp.where(
                mask,
                genes_mutated[indexes_parents],
                genes,
            )
            return genes_inherited

        new_batch_state_agents = tree_map(
            manage_genetic_component_inheritance,
            batch_state_agents,
            batch_state_agents_mutated,
        )

        # Agent-wise reaction
        key_random, subkey = random.split(key_random)
        new_batch_state_agents, batch_actions = self.act_agents(
            key_random=subkey,
            batch_observations=batch_observations,
            batch_state_agents=new_batch_state_agents,
        )

        return new_batch_state_agents, batch_actions

    # =============== Mutation methods =================

    def mutate_state_agent(
        self, state_agent: StateAgentEvolutionary, key_random: jnp.ndarray
    ) -> StateAgentEvolutionary:
        return state_agent.replace(
            age=0,
            params=mutation_gaussian_noise(
                arr=state_agent.params,
                strength_mutation=0.001,
                key_random=key_random,
            ),
        )

    # =============== Agent creation methods =================

    def act_agents(
        self,
        key_random: jnp.ndarray,
        batch_observations: ObservationAgent,  # Batched
        batch_state_agents: StateAgentEvolutionary,  # Batched
    ) -> jnp.ndarray:

        def act_single_agent(
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

        act_many_agents = jax.vmap(act_single_agent, in_axes=(0, 0, 0))
        batch_keys = random.split(key_random, self.n_agents_max)
        batch_state_agents, batch_actions = act_many_agents(
            batch_keys,
            batch_observations,
            batch_state_agents,
        )

        return batch_state_agents, batch_actions
