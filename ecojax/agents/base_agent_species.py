from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from flax import struct
import flax.linen as nn

from ecojax import spaces
from ecojax.core.eco_info import EcoInformation
from ecojax.models.base_model import BaseModel
from ecojax.spaces import EcojaxSpace
from ecojax.types import ActionAgent, ObservationAgent, StateSpecies


class AgentSpecies(ABC):
    """The base class for any species of agents.
    A species of agents represents a general method for eco-evolutionary algorithms environment-agnostic.

    It contains those two general notions :
    - How an agent reacts to its observation "o", where "o" is the observation received by the agent from the environment ?
    - What is the reproduction mecanism between agents indexes "i1", "i2", ..., "in", where those indexes are received from the environment ?

    For obeying the constant shape paradigm, even if the number of agents in the simulation is not constant, the number of "potential agents" is constant and equal to n_agents_max.
    This imply that any AgentSpecies instance should maintain a population of n_agents_max agents, even if some of them are not existing in the simulation at a given time.
    """
    
    def __init__(
        self,
        config: Dict,
        n_agents_max: int,
        n_agents_initial: int,
        observation_space: spaces.EcojaxSpace,
        action_space: spaces.DiscreteSpace,
        model_class: Type[BaseModel],
        config_model: Dict,
    ):
        self.config = config
        self.n_agents_max = n_agents_max
        self.n_agents_initial = n_agents_initial
        self.observation_space = observation_space
        self.action_space = action_space
        self.model_class = model_class
        self.config_model = config_model
        self.env = None
        

    @abstractmethod
    def reset(self, key_random: jnp.ndarray) -> StateSpecies:
        """Initialize the agents of the species. This should in particular initialize the agents species' state,
        i.e. the JAX dataclass that will contain the varying information of the agents species.

        Args:
            key_random (jnp.ndarray): the random key, of shape (2,)
        """

    @abstractmethod
    def react(
        self,
        state: StateSpecies,
        batch_observations: ObservationAgent,
        eco_information: EcoInformation,
        key_random: jnp.ndarray,
    ) -> Tuple[StateSpecies, ActionAgent, Dict[str, jnp.ndarray]]:
        """A function through which the agents reach to their observations and return their actions.
        It also handles the reproduction of the agents if required by the environment.

        Args:
            state (StateSpecies): the state of the species, as a StateSpecies object.
            batch_observations (jnp.ndarray): the observations, as a JAX structure of components, each of shape (n_agents_max, **dim_obs_component).
                It is composed of n_agents_max observations, each of them corresponding to the observation that the i-th indexed agent would receive.
            eco_information (EcoInformation): the eco-information of the environment, as an EcoInformation object.
            key_random (jnp.ndarray): the random key, of shape (2,)

        Returns:
            state_new (StateSpecies): the new state of the species, as a StateSpecies object.
            actions (jnp.ndarray): the actions of the agents, as a JAX array of shape (n_agents_max, **dim_action_component).
            info_species (Dict[str, jnp.ndarray]): a dictionary of additional information concerning the species of agents (e.g. metrics, etc.)
        """

    @abstractmethod
    def render(self, state: StateSpecies, force_render : bool = False) -> None:
        """Do the rendering of the species. This can be a visual rendering or a logging of the state of any kind.
        
        Args:
            state (StateSpecies): the state of the species to render
            force_render (bool): whether to force the rendering even if the species is not in a state where it should be rendered
        """
        return
    
    # ============== Helper methods ==============

    def compute_metrics(
        self,
        state: StateSpecies,
        state_new: StateSpecies,
        dict_measures: Dict[str, jnp.ndarray],
    ):

        # Set the measures to NaN for the agents that are not existing
        for name_measure, measures in dict_measures.items():
            name_measure_end = name_measure.split("/")[-1]
            if name_measure_end not in self.config["metrics"]["measures"]["global"]:
                dict_measures[name_measure] = jnp.where(
                    state_new.agents.do_exist,
                    measures,
                    jnp.nan,
                )

        # Aggregate the measures over the lifespan
        are_just_dead_agents = state_new.agents.do_exist & (
            ~state_new.agents.do_exist | (state_new.agents.age < state_new.agents.age)
        )

        dict_metrics_lifespan = {}
        new_list_metrics_lifespan = []
        for agg, metrics in zip(self.aggregators_lifespan, state.metrics_lifespan):
            new_metrics = agg.update_metrics(
                metrics=metrics,
                dict_measures=dict_measures,
                are_alive=state_new.agents.do_exist,
                are_just_dead=are_just_dead_agents,
                ages=state_new.agents.age,
            )
            dict_metrics_lifespan.update(agg.get_dict_metrics(new_metrics))
            new_list_metrics_lifespan.append(new_metrics)
        state_new_new = state_new.replace(metrics_lifespan=new_list_metrics_lifespan)

        # Aggregate the measures over the population
        dict_metrics_population = {}
        new_list_metrics_population = []
        dict_measures_and_metrics_lifespan = {**dict_measures, **dict_metrics_lifespan}
        for agg, metrics in zip(self.aggregators_population, state.metrics_population):
            new_metrics = agg.update_metrics(
                metrics=metrics,
                dict_measures=dict_measures_and_metrics_lifespan,
                are_alive=state_new.agents.do_exist,
                are_just_dead=are_just_dead_agents,
                ages=state_new.agents.age,
            )
            dict_metrics_population.update(agg.get_dict_metrics(new_metrics))
            new_list_metrics_population.append(new_metrics)
        state_new_new = state_new_new.replace(
            metrics_population=new_list_metrics_population
        )

        # Get the final metrics
        dict_metrics = {
            **dict_measures,
            **dict_metrics_lifespan,
            **dict_metrics_population,
        }

        # Arrange metrics in right format
        for name_metric in list(dict_metrics.keys()):
            *names, name_measure = name_metric.split("/")
            if len(names) == 0:
                name_metric_new = name_measure
            else:
                name_metric_new = f"{name_measure}/{' '.join(names[::-1])}"
            dict_metrics[name_metric_new] = dict_metrics.pop(name_metric)

        return state_new_new, dict_metrics
