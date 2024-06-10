from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.tree_util import register_pytree_node

from ecojax.metrics.base_aggregators import (
    Aggregator,
    AggregatorMeasureByMeasure,
    AggregatorNoMemory,
)


class AggregatorLifespanCumulative(AggregatorMeasureByMeasure):
    """Cumulative sum of the values of the measures over the lifespan of the agents.
    f(m)_t = sum_{i=1}^{t} m_i

    Should be applied only on immediate measures.
    """

    def __init__(
        self,
        config: Dict[str, Any],
    ):
        super().__init__(config=config)
        self.dict_cum_values = {
            f"{self.prefix_metric}/{key}": jnp.full((self.n_agents,), jnp.nan)
            for key in self.keys_measures
        }

    def aggregate_from_single_measure(
        self,
        name_measure: str,
        value_measure: jax.Array,
        are_alive: jax.Array,
        are_just_dead: jax.Array,
        ages: jax.Array,
    ) -> Dict[str, jax.Array]:
        name_metric = f"{self.prefix_metric}/{name_measure}"
        value_metric_last = self.dict_cum_values[name_metric]
        value_metric_new = jnp.select(
            condlist=[
                ~are_alive,
                ages == 1,
            ],
            choicelist=[
                jnp.nan,
                value_measure,
            ],
            default=value_metric_last + value_measure,
        )
        self.dict_cum_values[name_metric] = value_metric_new
        return {name_metric: value_metric_new}

    def tree_flatten(self):
        leaves = list(self.dict_cum_values.values())
        aux_data = self.config
        return leaves, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        config = aux_data
        agg = cls(config=config)
        agg.dict_cum_values = {
            f"{config['prefix_metric']}/{key}": leaves[i]
            for i, key in enumerate(config["keys_measures"])
        }
        return agg


class AggregatorLifespanAverage(AggregatorMeasureByMeasure):
    """Time-average of the values of the measures over the lifespan of the agents.
    f(m)_t = sum_{i=1}^{t} m_i / t

    Can be applied to any measure.
    """

    def __init__(
        self,
        config: Dict[str, Any],
    ):
        super().__init__(config=config)
        self.dict_avg_values = {
            f"{self.prefix_metric}/{key}": jnp.full((self.n_agents,), jnp.nan)
            for key in self.keys_measures
        }

    def aggregate_from_single_measure(
        self,
        name_measure: str,
        value_measure: jax.Array,
        are_alive: jax.Array,
        are_just_dead: jax.Array,
        ages: jax.Array,
    ) -> Dict[str, jax.Array]:
        name_metric = f"{self.prefix_metric}/{name_measure}"
        value_metric_last = self.dict_avg_values[name_metric]
        value_metric_new = jnp.select(
            condlist=[
                ~are_alive,
                ages == 1,
            ],
            choicelist=[
                jnp.nan,
                value_measure,
            ],
            default=(value_metric_last * (ages - 1) + value_measure) / ages,
        )
        self.dict_avg_values[name_metric] = value_metric_new
        return {name_metric: value_metric_new}

    def tree_flatten(self):
        leaves = list(self.dict_avg_values.values())
        aux_data = self.config
        return leaves, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        config = aux_data
        agg = cls(config=config)
        agg.dict_avg_values = {
            f"{config['prefix_metric']}/{key}": children[i]
            for i, key in enumerate(config["keys_measures"])
        }
        return agg


class AggregatorLifespanIncrementalChange(AggregatorMeasureByMeasure):
    """Incremental change of the values of the measures over the lifespan of the agents.
    f(m)_t = m_t - m_{t-1}

    Should be applied only on state/behavior measures.
    """

    def __init__(
        self,
        config: Dict[str, Any],
    ):
        super().__init__(config=config)
        self.dict_last_values = {
            key: jnp.full((self.n_agents,), jnp.nan) for key in self.keys_measures
        }

    def aggregate_from_single_measure(
        self,
        name_measure: str,
        value_measure: jax.Array,
        are_alive: jax.Array,
        are_just_dead: jax.Array,
        ages: jax.Array,
    ) -> Dict[str, jax.Array]:
        name_metric = f"{self.prefix_metric}/{name_measure}"
        value_measure_last = self.dict_last_values[name_measure]
        # Compute the new metric : value - value_last
        value_metric_new = jnp.select(
            condlist=[
                ~are_alive,
                ages == 1,
            ],
            choicelist=[
                jnp.nan,
                jnp.nan,
            ],
            default=value_measure - value_measure_last,
        )
        # Update the new metric
        self.dict_last_values[name_measure] = value_measure
        return {name_metric: value_metric_new}

    def tree_flatten(self):
        leaves = list(self.dict_last_values.values())
        aux_data = self.config
        return leaves, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        config = aux_data
        agg = cls(config=config)
        agg.dict_last_values = {
            key: children[i] for i, key in enumerate(config["keys_measures"])
        }
        return agg


class AggregatorLifespanAbsoluteChange(AggregatorMeasureByMeasure):
    """Difference of values of the measures over the lifespan of the agents.
    f(m)_t = m_t - m_1

    Should be applied only on state/behavior measures.
    """

    def __init__(
        self,
        config: Dict[str, Any],
    ):
        super().__init__(config=config)
        self.dict_first_values = {
            key: jnp.full((self.n_agents,), jnp.nan) for key in self.keys_measures
        }

    def aggregate_from_single_measure(
        self,
        name_measure: str,
        value_measure: jax.Array,
        are_alive: jax.Array,
        are_just_dead: jax.Array,
        ages: jax.Array,
    ) -> Dict[str, jax.Array]:
        name_metric = f"{self.prefix_metric}/{name_measure}"
        # Update the first value (in case ages==1)
        self.dict_first_values[name_measure] = jnp.select(
            condlist=[
                ~are_alive,
                ages == 1,
            ],
            choicelist=[
                jnp.nan,
                value_measure,
            ],
            default=self.dict_first_values[name_measure],
        )
        first_value = self.dict_first_values[name_measure]
        # Compute the new metric : value - value_last
        value_metric_new = jnp.select(
            condlist=[
                ~are_alive,
                ages == 1,
            ],
            choicelist=[
                jnp.nan,
                0,
            ],
            default=value_measure - first_value,
        )
        return {name_metric: value_metric_new}

    def tree_flatten(self):
        leaves = list(self.dict_first_values.values())
        aux_data = self.config
        return leaves, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        config = aux_data
        agg = cls(config=config)
        agg.dict_first_values = {
            key: children[i] for i, key in enumerate(config["keys_measures"])
        }
        return agg


class AggregatorLifespanFinal(AggregatorMeasureByMeasure, AggregatorNoMemory):
    """Final value of the measures over the lifespan of the agents.

    f(m)_t = {
        m_T if t==T
        NaN else
        }

    Should be only applied to state/behavior measures.
    """

    def __init__(self, config: Dict[str, Any]):
        assert (not "log_final" in config) or (
            not config["log_final"]
        ), "log_final should be false, as this aggregator already log the final value"
        super().__init__(config)

    def aggregate_from_single_measure(
        self,
        name_measure: str,
        value_measure: jax.Array,
        are_alive: jax.Array,
        are_just_dead: jax.Array,
        ages: jax.Array,
    ) -> Dict[str, jax.Array]:
        return {name_measure: jnp.where(are_just_dead, value_measure, jnp.nan)}


# ================================ Population metrics ================================
class AggregatorPopulationMean(AggregatorMeasureByMeasure, AggregatorNoMemory):
    """Sum of the values of the measures over the population.

    f(m) = mean(m)

    Can be applied to any measure, thanks to the linearity of mean.
    """

    def aggregate_from_single_measure(
        self,
        name_measure: str,
        value_measure: jax.Array,
        are_alive: jax.Array,
        are_just_dead: jax.Array,
        ages: jax.Array,
    ) -> Dict[str, jax.Array]:
        return {f"{self.prefix_metric}/{name_measure}": jnp.nanmean(value_measure)}


class AggregatorPopulationStd(AggregatorMeasureByMeasure, AggregatorNoMemory):
    """Standard deviation of the values of the measures over the population.

    f(m) = std(m)

    Can be applied to any measure, but makes more sense for state/behavior measure and
    life-aggregated metrics.
    """

    def aggregate_from_single_measure(
        self,
        name_measure: str,
        value_measure: jax.Array,
        are_alive: jax.Array,
        are_just_dead: jax.Array,
        ages: jax.Array,
    ) -> Dict[str, jax.Array]:
        return {f"{self.prefix_metric}/{name_measure}": jnp.nanstd(value_measure)}


class AggregatorPopulationMovingMean(AggregatorMeasureByMeasure):
    """Mean of the values of the measures over the population, but not only on the current
    timestep, but also to previous timesteps.
    """

    def __init__(
        self,
        config: Dict[str, Any],
    ):
        super().__init__(config=config)
        self.learning_rate = config["learning_rate"]
        self.dict_moving_average = {
            f"{self.prefix_metric}/{key}": jnp.full((self.n_agents,), jnp.nan)
            for key in self.keys_measures
        }

    def aggregate_from_single_measure(
        self,
        name_measure: str,
        value_measure: jax.Array, # (n,) shape
        are_alive: jax.Array,
        are_just_dead: jax.Array,
        ages: jax.Array,
    ) -> Dict[str, jax.Array]:
        name_metric = f"{self.prefix_metric}/{name_measure}"
        value_metric_last = self.dict_moving_average[name_metric]
        value_metric_new = jnp.where(
            value_metric_last == jnp.nan,
            value_measure,
            value_metric_last + self.learning_rate * (value_measure - value_metric_last),
            )
        self.dict_moving_average[name_metric] = value_metric_new
        return {name_metric: value_metric_new}

    def tree_flatten(self):
        leaves = list(self.dict_moving_average.values())
        aux_data = self.config
        return leaves, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        config = aux_data
        agg = cls(config=config)
        agg.dict_moving_average = {
            f"{config['prefix_metric']}/{key}": children[i]
            for i, key in enumerate(config["keys_measures"])
        }
        return agg


# Register the Pytree nodes
list_Aggregator: List[Type[Aggregator]] = [
    # Lifespan aggregator
    AggregatorLifespanCumulative,
    AggregatorLifespanAverage,
    AggregatorLifespanIncrementalChange,
    AggregatorLifespanAbsoluteChange,
    AggregatorLifespanFinal,
    # Population aggregator
    AggregatorPopulationMean,
    AggregatorPopulationMovingMean,
    AggregatorPopulationStd,
]

for AggregatorClass in list_Aggregator:
    register_pytree_node(
        AggregatorClass,
        flatten_func=AggregatorClass.tree_flatten,
        unflatten_func=AggregatorClass.tree_unflatten,
    )
