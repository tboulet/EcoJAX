from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Tuple, Type, Union

import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from flax import struct
import flax.linen as nn

from ecojax.core.eco_info import EcoInformation
from ecojax.models.base_model import BaseModel
from ecojax.spaces import EcojaxSpace
from ecojax.types import ObservationAgent, StateSpecies


class AgentSpecies(ABC):
    """The base class for any species of agents.
    A species of agents represents a general method for eco-evolutionary algorithms environment-agnostic.

    It contains those two general notions :
    - How an agent reacts to its observation "o", where "o" is the observation received by the agent from the environment ?
    - What is the reproduction mecanism between agents indexes "i1", "i2", ..., "in", where those indexes are received from the environment ?

    For obeying the constant shape paradigm, even if the number of agents in the simulation is not constant, the number of "potential agents" is constant and equal to n_agents_max.
    This imply that any AgentSpecies instance should maintain a population of n_agents_max agents, even if some of them are not existing in the simulation at a given time.
    """

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
    ) -> jnp.ndarray:
        """A function through which the agents reach to their observations and return their actions.
        It also handles the reproduction of the agents if required by the environment.

        Args:
            state (StateSpecies): the state of the species, as a StateSpecies object.
            batch_observations (jnp.ndarray): the observations, as a JAX structure of components, each of shape (n_agents_max, **dim_obs_component).
                It is composed of n_agents_max observations, each of them corresponding to the observation that the i-th indexed agent would receive.
            eco_information (EcoInformation): the eco-information of the environment, as an EcoInformation object.
            key_random (jnp.ndarray): the random key, of shape (2,)

        Returns:
            action (int): the action of the agent, as an integer
        """

    # ============== Helper methods ==============

    def compute_metrics(
        self,
        state: StateSpecies,
        state_new: StateSpecies,
        dict_measures: Dict[str, jnp.ndarray],
    ):

        # Set the measures to NaN for the agents that are not existing
        for name_measure, measures in dict_measures.items():
            if name_measure not in self.config["metrics"]["measures"]["global"]:
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
