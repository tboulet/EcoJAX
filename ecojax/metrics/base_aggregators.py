from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.tree_util import register_pytree_node

from ecojax.types import PytreeLike
from ecojax.utils import jprint_and_breakpoint


class Aggregator(PytreeLike):
    """The base class for all aggregators of a certain measure over the lifespan of an agent.
    At age t, the metric represented by any instance of this class is an aggregation of the measures of the agent from age 0 to t.
    To do that, an aggregator apply the function aggregate() at each timestep, which updates the (sometimes necessary) internal state of the aggregator and return the aggregated measures.

    The measure that is given as input, as well as the are_alive and ages arrays, are the measures of the agents at the beginning of the timestep (or as a consequence of their action).
    They don't concern the newborn agents, which are not yet active.

    Rules:
        1: When the agent is not alive, it means the agent was dead at beginning of the timestep. The aggregator should
            consequently return NaN in most cases.
        2: When age==1, this can be considered as the first active timestep of the agent. This will be considered as the
            initialization of most aggregators.
        3: Otherwise, it means the agent was already active in the previous timestep. The aggregator should update the
            internal state of the aggregator and return the aggregated measures.
    """

    def __init__(
        self,
        config: Dict[str, Any],
    ):
        """Initialize the aggregator.

        Args:
            config (Dict[str, Any]): the configuration of the aggregator
        """
        self.config = config
        self.keys_measures: List[str] = config["keys_measures"]
        self.keys_measures_prefix: List[str] = config.get("keys_measures_prefix", [])
        self.n_agents: str = config["n_agents"]
        self.prefix_metric: str = config.get("prefix_metric", None)
        self.log_final: bool = config.get("log_final", False)

    @abstractmethod
    def aggregate(
        self,
        dict_measures: Dict[str, jnp.ndarray],
        are_alive: jnp.ndarray,
        are_just_dead: jnp.ndarray,
        ages: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray]:
        """Aggregate the new measure (for all agents and all measure types) on it's lifespan.

        Args:
            dict_measures (Dict[str, jnp.ndarray]): the measure dictionnary, as a mapping from measure name k to measure values (m_t^(i,k))_i of a certain shape
            are_alive (jnp.ndarray): an array of booleans of shape (n_agents,) indicating if the agents are alive
            are_just_dead (jnp.ndarray): an array of booleans of shape (n_agents,) indicating if the agents are just dead
            ages (jnp.ndarray): the age of the agents, as an array of integers of shape (n_agents,) indicating the age of the agents

        Returns:
            Dict[str, jnp.ndarray]: the aggregated measures, as a mapping from metric name k' to aggregated measure values of a certain shape
        """


class AggregatorMeasureByMeasure(Aggregator):
    """An util cass that can be subclassed whenever you want to define an aggregator
    that aggregates metrics dictionnaries measure-independently, which is often the case.
    """

    @abstractmethod
    def aggregate_from_single_measure(
        self,
        name_measure: str,
        value_measure: jax.Array,
        are_alive: jax.Array,
        are_just_dead: jax.Array,
        ages: jax.Array,
    ) -> Dict[str, jax.Array]:
        """Return the aggregated metric(s) obtained from a single measure
        Example of use :

        Mean of values :
        >>> def aggregate_from_single_measure(self, name_measure, value, are_alive, are_just_dead, ages):
        >>>    return {f"{self.prefix_metric}/{name_measure}": jnp.nanmean(value)}

        Cumulative sum of values :
        >>> def aggregate_from_single_measure(self, name_measure, value, are_alive, are_just_dead, ages):
        >>>    return {f"{self.prefix_metric}/{name_measure}": value}

        Args:
            name_measure (str): the name of the measure
            value_measure (jax.Array): the value of the measure
            are_alive (jax.Array): the array of booleans indicating if the agents are alive, of shape (n_agents,)
            are_just_dead (jax.Array): the array of booleans indicating if the agents are just dead, of shape (n_agents,)
            ages (jax.Array): the age of the agents, of shape (n_agents,)

        Returns:
            Dict[str, jax.Array]: the aggregated metric(s)
        """
        raise NotImplementedError

    def aggregate(
        self,
        dict_measures: Dict[str, jnp.ndarray],
        are_alive: jnp.ndarray,
        are_just_dead: jnp.ndarray,
        ages: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray]:
        dict_metrics_aggregated = {}
        for name_measure, value_measure in dict_measures.items():
            # Check if name measure is either in keys_measures or has a prefix in keys_measures_prefix, in this case, aggregate it
            if (name_measure in self.keys_measures) or any(
                [
                    name_measure.startswith(prefix_measure)
                    for prefix_measure in self.keys_measures_prefix
                ]
            ):
                # Add the aggregated metrics to the dictionnary using the method aggregate_from_single_measure
                dict_metrics_aggregated_from_measure = (
                    self.aggregate_from_single_measure(
                        name_measure=name_measure,
                        value_measure=value_measure,
                        are_alive=are_alive,
                        are_just_dead=are_just_dead,
                        ages=ages,
                    )
                )
                dict_metrics_aggregated.update(dict_metrics_aggregated_from_measure)
        return dict_metrics_aggregated


class AggregatorNoMemory(Aggregator):
    """A class that implements trivially the PytreeLike required abstract methods.
    This class can be subclassed when needing to create a pytree like object with no memory, ie whose output is only based on current measure.
    """

    def tree_flatten(self):
        leaves = ()
        aux_data = self.config
        return leaves, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(aux_data)
