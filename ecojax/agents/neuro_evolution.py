from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Type

import jax
from jax import random, tree_map, tree_structure
import jax.numpy as jnp
import numpy as np
from flax.struct import PyTreeNode, dataclass
import flax.linen as nn


from ecojax.agents.base_agent_species import AgentSpecies
from ecojax.core.eco_info import EcoInformation
from ecojax.metrics.aggregators import Aggregator
from ecojax.models.base_model import BaseModel
from ecojax.evolution.mutator import mutate_scalar, mutation_gaussian_noise
from ecojax.types import ActionAgent, ObservationAgent
import ecojax.spaces as spaces
from ecojax.utils import instantiate_class, jprint, get_dict_flattened


@dataclass
class HyperParametersNE:
    # The mutation strength of the agent
    strength_mutation: float


@dataclass
class AgentNE:
    # The age of the agent, in number of timesteps
    age: int
    # The parameters of the neural network corresponding to the agent
    params: Dict[str, jnp.ndarray]
    # Hyperparameters of the agent
    hp: HyperParametersNE
    # Whether the agent is existing
    do_exist: bool


@dataclass
class StateSpeciesNE:
    # The agents of the species
    agents: AgentNE
    # The lifespan and population aggregators
    metrics_lifespan: List[PyTreeNode]
    metrics_population: List[PyTreeNode]


class NeuroEvolutionAgentSpecies(AgentSpecies):
    """A species of agents that evolve their neural network weights."""

    def __init__(
        self,
        config: Dict,
        n_agents_max: int,
        n_agents_initial: int,
        observation_space: spaces.EcojaxSpace,
        n_actions: int,
        model_class: Type[BaseModel],
        config_model: Dict,
    ):
        self.config = config
        self.n_agents_max = n_agents_max
        self.n_agents_initial = n_agents_initial
        self.observation_space = observation_space
        self.n_actions = n_actions
        
        # Model
        self.model = model_class(
            space_input=observation_space,
            space_output=spaces.ContinuousSpace(shape=(n_actions,)),
            **config_model,
        )
        print(f"Model: {self.model.get_table_summary()}")
        
        # Metrics parameters
        self.names_measures: List[str] = sum(
            [names for type_measure, names in config["metrics"]["measures"].items()], []
        )

    def reset(self, key_random: jnp.ndarray) -> StateSpeciesNE:

        # Initialize the metrics
        self.aggregators_lifespan: List[Aggregator] = []
        list_metrics_lifespan: List[PyTreeNode] = []
        for config_agg in self.config["metrics"]["aggregators_lifespan"]:
            agg: Aggregator = instantiate_class(**config_agg)
            self.aggregators_lifespan.append(agg)
            list_metrics_lifespan.append(agg.get_initial_metrics())

        self.aggregators_population: List[Aggregator] = []
        list_metrics_population: List[PyTreeNode] = []
        for config_agg in self.config["metrics"]["aggregators_population"]:
            agg: Aggregator = instantiate_class(**config_agg)
            self.aggregators_population.append(agg)
            list_metrics_population.append(agg.get_initial_metrics())

        # Initialize the state
        key_random, subkey = random.split(key_random)
        batch_keys = jax.random.split(subkey, self.n_agents_max)
        batch_agents = jax.vmap(self.init_agent)(
            key_random=batch_keys,
            do_exist=jnp.arange(self.n_agents_max)
            < self.n_agents_initial,  # by convention, the first n_agents_initial agents exist
        )
        return StateSpeciesNE(
            agents=batch_agents,
            metrics_lifespan=list_metrics_lifespan,
            metrics_population=list_metrics_population,
        )

    def init_hp(self) -> HyperParametersNE:
        """Get the initial hyperparameters of the agent from the config"""
        return HyperParametersNE(**self.config["hp_initial"])

    def init_agent(
        self,
        key_random: jnp.ndarray,
        do_exist: bool = True,
        params: Optional[Dict[str, jnp.ndarray]] = None,
        hp: Optional[HyperParametersNE] = None,
    ) -> AgentNE:
        """Create a new agent.

        Args:
            key_random (jnp.ndarray): the random key
            do_exist (bool, optional): whether the agent exists. Defaults to True.
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
            do_exist=True,
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
        Dict[str, jnp.ndarray],
    ]:

        # Initialize the measures dictionnary. This will be used to store the measures of the environment at this step.
        dict_measures_all: Dict[str, jnp.ndarray] = {}

        # Apply the mutation
        batch_keys = random.split(key_random, self.n_agents_max)
        agents_mutated = jax.vmap(self.mutate_state_agent)(
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

        new_agents = tree_map(
            manage_genetic_component_inheritance,
            state.agents,
            agents_mutated,
        )

        # Update the do_exist attribute
        new_do_exist = (
            new_agents.do_exist & ~eco_information.are_just_dead_agents
        )  # Remove the agents that have just died
        new_do_exist = new_do_exist | are_newborns_agents  # Add the newborn agents
        new_agents = new_agents.replace(do_exist=new_do_exist)

        new_state = state.replace(agents=new_agents)

        # ================== Agent-wise reaction ==================
        key_random, subkey = random.split(key_random)
        new_state, batch_actions, dict_measures = self.react_agents(
            key_random=subkey,
            state=new_state,
            batch_observations=batch_observations,
        )
        dict_measures_all.update(dict_measures)

        # ============ Compute the metrics ============

        # Compute some measures
        key_random, subkey = random.split(key_random)
        dict_measures = self.compute_measures(
            state=state, state_new=new_state, key_random=subkey
        )
        dict_measures_all.update(dict_measures)

        # Set the measures to NaN for the agents that are not existing
        for name_measure, measures in dict_measures_all.items():
            if name_measure not in self.config["metrics"]["measures"]["global"]:
                dict_measures_all[name_measure] = jnp.where(
                    new_state.agents.do_exist,
                    measures,
                    jnp.nan,
                )

        # Update and compute the metrics
        new_state, dict_metrics = self.compute_metrics(
            state=state, state_new=new_state, dict_measures=dict_measures_all
        )
        
        # Also log the weights of one agent
        params_agent0 = tree_map(
            f=lambda x: x[0].reshape(-1), # get the first element of the array and flatten it
            tree=state.agents.params,
        )
        params_flattened_agent0 = get_dict_flattened(d=params_agent0, sep=' ')
        for key, value in params_flattened_agent0.items():
            dict_metrics[f"params_agent0/{key}"] = value
        
        info = {"metrics": dict_metrics}

        return new_state, batch_actions, info

    # =============== Mutation methods =================

    def mutate_state_agent(self, agent: AgentNE, key_random: jnp.ndarray) -> AgentNE:
        # Mutate the hyperparameters
        key_random, *subkeys = random.split(key_random, 5)
        new_hp = HyperParametersNE(
            strength_mutation=mutate_scalar(
                value=agent.hp.strength_mutation, range=(0, None), key_random=subkeys[3]
            ),
        )
        # Transmit the initial weights, slightly mutated
        key_random, subkey = random.split(key_random)
        new_params = mutation_gaussian_noise(
            arr=agent.params,
            strength_mutation=agent.hp.strength_mutation,
            key_random=subkey,
        )

        key_random, *subkeys = random.split(key_random, 5)
        return agent.replace(
            age=0,
            params=new_params,
            hp=new_hp,
        )

    # =============== Agent creation methods =================

    def react_single_agent(
            self,
            key_random: jnp.ndarray,
            agent: AgentNE,
            obs: jnp.ndarray,
        ) -> Tuple[
            AgentNE,
            ActionAgent,
            Dict[str, jnp.ndarray],
        ]:
        # =============== Inference part =================
        key_random, subkey = random.split(key_random)
        logits = self.model.apply(
            variables={"params": agent.params},
            x=obs,
            key_random=subkey,
        )
        key_random, subkey = random.split(key_random)
        action = random.categorical(key_random, logits=logits)
        # ============== Learning part (no learning in NE) =================
        agent.replace(age=agent.age + 1)
        # ============== Measures ==============
        probs = jax.nn.softmax(logits)
        dict_measures = {
            "prob_max": jnp.max(probs),
            "prob_min": jnp.min(probs),
            "prob_median": jnp.median(probs),
            "entropy": -jnp.sum(probs * jnp.log(probs)),
        }
        # Update the agent's state and act
        return agent, action, dict_measures
        
        
    def react_agents(
        self,
        key_random: jnp.ndarray,
        state: StateSpeciesNE,
        batch_observations: ObservationAgent,  # Batched
    ) -> Tuple[
        StateSpeciesNE,
        ActionAgent,  # Batched
    ]:
        batch_keys = random.split(key_random, self.n_agents_max)
        new_agents, actions, dict_measures = jax.vmap(self.react_single_agent)(
            key_random=batch_keys,
            agent=state.agents,
            obs=batch_observations,
        )

        new_state = state.replace(agents=new_agents)
        return new_state, actions, dict_measures

    # =============== Metrics methods =================

    def compute_measures(
        self,
        state: StateSpeciesNE,
        state_new: StateSpeciesNE,
        key_random: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray]:
        """Compute the measures of the agents."""
        dict_measures = {}
        for name_measure in self.names_measures:
            # Global measures
            pass
            # Immediate measures
            pass
            # State measures
            if "hp" in name_measure:
                strength_mutation = getattr(state.agents.hp, "strength_mutation")
                dict_measures["strength_mutation"] = strength_mutation
                dict_measures["log10/strength_mutation"] = jnp.log10(strength_mutation)
            # Behavior measures
            pass

        return dict_measures
