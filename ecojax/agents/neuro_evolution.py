from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, List, Tuple, Type

import jax
from jax import random, tree_map
import jax.numpy as jnp
import numpy as np
from flax import struct
import flax.linen as nn


from ecojax.agents.base_agent_species import AgentSpecies
from ecojax.core.eco_info import EcoInformation
from ecojax.models.base_model import BaseModel
from ecojax.evolution.mutator import mutate_scalar, mutation_gaussian_noise
from ecojax.types import ActionAgent, ObservationAgent
import ecojax.spaces as spaces
from ecojax.utils import jprint


@struct.dataclass
class HyperParametersNE:
    # The mutation strength of the agent
    strength_mutation: float


@struct.dataclass
class AgentNE:
    # The age of the agent, in number of timesteps
    age: int

    # The parameters of the neural network corresponding to the agent
    params: Dict[str, jnp.ndarray]

    # Hyperparameters of the agent
    hp: HyperParametersNE


@struct.dataclass
class StateSpeciesNE:
    # The agents of the species
    agents: AgentNE


class NeuroEvolutionAgentSpecies(AgentSpecies):
    """A species of agents that evolve their neural network weights."""

    mode_return: str = "action_prob"
    
    def reset(self, key_random: jnp.ndarray) -> StateSpeciesNE:
        # Initialize the state
        key_random, subkey = random.split(key_random)
        batch_keys = jax.random.split(subkey, self.n_agents_max)
        batch_agents = jax.vmap(self.init_agent)(batch_keys)
        return StateSpeciesNE(
            agents=batch_agents,
        )

    def init_hp(self) -> HyperParametersNE:
        """Get the initial hyperparameters of the agent from the config"""
        return HyperParametersNE(**self.config["hp_initial"])

    def init_agent(
        self,
        key_random: jnp.ndarray,
        params: Dict[str, jnp.ndarray] = None,
        hp: HyperParametersNE = None,
    ) -> AgentNE:
        """Create a new agent.

        Args:
            key_random (jnp.ndarray): the random key
            params (Dict[str, jnp.ndarray], optional): the NN parameters. Defaults to None (will be initialized randomly)
            hp (HyperParametersRL, optional): the hyperparameters of the agent. Defaults to None (will be initialized from the config).

        Returns:
            AgentRL: the new agent
        """
        if params is None:
            variables = self.model.get_initialized_variables(key_random)
            params = variables.get("params", {})
        if hp is None:
            hp = self.init_hp()
        return AgentNE(
            age=0,
            params=params,
            hp=hp,
        )
        
    def react(
        self,
        state: StateSpeciesNE,
        batch_observations: ObservationAgent,  # Batched
        eco_information: EcoInformation,
        key_random: jnp.ndarray,
    ) -> Tuple[
        StateSpeciesNE,
        ActionAgent,  # Batched
    ]:

        # Apply the mutation
        batch_keys = random.split(key_random, self.n_agents_max)
        batch_agents_mutated = jax.vmap(self.mutate_state_agent)(
            agent=state.agents, key_random=batch_keys
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

        new_batch_agents = tree_map(
            manage_genetic_component_inheritance,
            state.agents,
            batch_agents_mutated,
        )

        new_state_species = StateSpeciesNE(
            agents=new_batch_agents,
        )

        # Agent-wise reaction
        key_random, subkey = random.split(key_random)
        new_state_species, batch_actions = self.act_agents(
            key_random=subkey,
            state_species=new_state_species,
            batch_observations=batch_observations,
        )

        return new_state_species, batch_actions


    # =============== Mutation methods =================

    def mutate_state_agent(
        self, agent: AgentNE, key_random: jnp.ndarray
    ) -> AgentNE:
        key_random, *subkeys = random.split(key_random, 5)
        return agent.replace(
            age=0,
            params=mutation_gaussian_noise(
                arr=agent.params,
                strength_mutation=agent.hp.strength_mutation,
                key_random=key_random,
            ),
            hp=HyperParametersNE(strength_mutation=mutate_scalar(
                value=agent.hp.strength_mutation, range=(0, None), key_random=subkeys[3]
            ))
        )

    # =============== Agent creation methods =================

    def act_agents(
        self,
        key_random: jnp.ndarray,
        state_species: StateSpeciesNE,
        batch_observations: ObservationAgent,  # Batched
    ) -> jnp.ndarray:

        def act_single_agent(
            key_random: jnp.ndarray,
            state_agent: AgentNE,
            obs: jnp.ndarray,
        ) -> jnp.ndarray:
            # Inference part
            action, prob_action = self.model.apply(
                variables={"params": state_agent.params},
                obs=obs,
                key_random=key_random,
                mode_return="action_prob",
            )
            # Learning part
            state_agent.replace(age=state_agent.age + 1)
            pass  # TODO: Implement the learning part
            # Update the agent's state and act
            return state_agent, action

        batch_keys = random.split(key_random, self.n_agents_max)
        batch_agents, batch_actions = jax.vmap(act_single_agent)(
            key_random=batch_keys,
            state_agent=state_species.agents,
            obs=batch_observations,
        )

        new_state_species = StateSpeciesNE(
            agents=batch_agents,
        )
        return new_state_species, batch_actions
