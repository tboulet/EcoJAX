from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.tree_util import register_pytree_node

from ecojax.types import PytreeLike


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
        for key, value in config.items():
            setattr(self, key, value)
        # if "keys_measures_prefix" in config:
        #     for key_measure_pr in self.keys_measures_prefix:
        #         self.keys_measures += [
        #             f"{key_measure_pr}_{key}" for key in self.keys_measures
        #         ]

    @abstractmethod
    def aggregate(
        self,
        dict_measures: Dict[str, jnp.ndarray],
        are_alive: jnp.ndarray,
        ages: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray]:
        """Aggregate the new measure (for all agents and all measure types) on it's lifespan.

        Args:
            dict_measures (Dict[str, jnp.ndarray]): the measure dictionnary, as a mapping from measure name k to measure values (m_t^(i,k))_i of shape (n_agents,)
            are_alive (jnp.ndarray): an array of booleans of shape (n_agents,) indicating if the agents are alive
            ages (jnp.ndarray): the age of the agents, as an array of integers of shape (n_agents,) indicating the age of the agents

        Returns:
            Dict[str, jnp.ndarray]: the aggregated measures, as a mapping from measure name k to aggregated measure values of shape (n,) or ()
        """


class AggregatorLifespanCumulative(Aggregator):

    def __init__(
        self,
        config: Dict[str, Any],
    ):
        super().__init__(config=config)
        self.dict_cum_values = {
            f"{self.prefix_metric}_{key}": jnp.full((self.n_agents,), jnp.nan)
            for key in self.keys_measures
        }

    def aggregate(
        self,
        dict_measures: Dict[str, jnp.ndarray],
        are_alive: jnp.ndarray,
        ages: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray]:

        for key in self.keys_measures:
            assert key in dict_measures, f"Key {key} not in dict_measures"
            m = dict_measures[key]
            self.dict_cum_values[f"{self.prefix_metric}_{key}"] = jnp.select(
                condlist=[
                    ~are_alive,
                    ages == 1,
                ],
                choicelist=[
                    jnp.nan,
                    m,
                ],
                default=self.dict_cum_values[f"{self.prefix_metric}_{key}"] + m,
            )
        return self.dict_cum_values

    def tree_flatten(self):
        leaves = list(self.dict_cum_values.values())
        aux_data = self.config
        return leaves, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        config = aux_data
        agg = cls(config=config)
        agg.dict_cum_values = {
            f"{config['prefix_metric']}_{key}": children[i]
            for i, key in enumerate(config["keys_measures"])
        }
        return agg


class AggregatorPopulationMean(Aggregator):

    def aggregate(self, dict_metrics: Dict[str, jax.Array]) -> Dict[str, jax.Array]:
        dict_metrics_aggregated = {}
        for name_metric, value in dict_metrics.items():
            name_metric_aggregated = f"{self.prefix_metric}_{name_metric}"
            if name_metric in self.keys_measures:
                dict_metrics_aggregated[name_metric_aggregated] = jnp.nanmean(value)
            else:
                for prefix_measure in self.keys_measures_prefix:
                    if name_metric.startswith(prefix_measure):
                        dict_metrics_aggregated[name_metric_aggregated] = jnp.nanmean(value)
                        break
        return dict_metrics_aggregated


    def tree_flatten(self):
        return (), self.config

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(aux_data)


# Register the Pytree nodes
list_Aggregator: List[Type[Aggregator]] = [
    AggregatorLifespanCumulative,
    AggregatorPopulationMean,
]

for AggregatorClass in list_Aggregator:
    register_pytree_node(
        AggregatorClass,
        flatten_func=AggregatorClass.tree_flatten,
        unflatten_func=AggregatorClass.tree_unflatten,
    )
