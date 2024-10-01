from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.tree_util import register_pytree_node
from flax.struct import PyTreeNode, dataclass


class Metric(PyTreeNode):
    """A class to represent a metric. It should be a PyTreeNode."""


class Aggregator(ABC):

    def __init__(self, config: Dict[str, Any]):
        self.config: Dict[str, Any] = config
        self.keys_measures: List[str] = config["keys_measures"]
        self.keys_measures_prefix: List[str] = config.get("keys_measures_prefix", [])
        self.n_agents: str = config["n_agents"]
        self.prefix_metric: str = config["prefix_metric"]
        self.log_final: bool = config.get("log_final", False)

    @abstractmethod
    def get_initial_metrics(self) -> Metric:
        """Return the initial metric object of the aggregator."""
        raise NotImplementedError

    @abstractmethod
    def update_metrics(
        self,
        metrics: Metric,
        dict_measures: Dict[str, jnp.ndarray],
        are_alive: jnp.ndarray,
        are_just_dead: jnp.ndarray,
        ages: jax.Array,
    ) -> Metric:
        """Return the updated metrics of the aggregator."""
        raise NotImplementedError

    @abstractmethod
    def get_dict_metrics(self, metrics: Metric) -> Dict[str, jnp.ndarray]:
        """Return the metrics of the aggregator as a dictionnary."""
        return {}

    def get_dict_of_full_arrays(
        self, fill_value: jnp.ndarray, mode: str = "histogram"
    ) -> Dict[str, jnp.ndarray]:
        """Return a dictionnary of arrays, whose keys are based on the keys_measures and the prefix_metric attribute,
        and whose values are full arrays of a shape depending on the mode argument.

        Args:
            fill_value (jnp.ndarray): the value to fill the arrays with
            mode (str): the mode of the fill value, either "histogram" or "scalar"
        Returns:
            Dict[str, jnp.ndarray]: a dictionnary of arrays
        """
        if mode == "histogram":
            return {
                f"{self.prefix_metric}/{name_measure}": jnp.full(
                    (self.n_agents,), fill_value
                )
                for name_measure in self.keys_measures
            }
        elif mode == "scalar":
            return {
                f"{self.prefix_metric}/{name_measure}": fill_value
                for name_measure in self.keys_measures
            }
        else:
            raise ValueError(f"Unknown mode {mode}.")


# ================================ Population metrics ================================
class AggregatorPopulationMean(Aggregator):
    """Sum of the values of the measures over the population.

    f(m) = mean(m)

    Can be applied to any measure, thanks to the linearity of mean.
    """

    def get_initial_metrics(self) -> Metric:
        return self.get_dict_of_full_arrays(fill_value=jnp.nan, mode="scalar")

    def update_metrics(
        self,
        metrics: Metric,
        dict_measures: Dict[str, jnp.ndarray],
        are_alive: jnp.ndarray,
        are_just_dead: jnp.ndarray,
        ages: jax.Array,
    ) -> Metric:
        dict_metrics_aggregated = {}
        for name_measure in dict_measures.keys():
            if (name_measure in self.keys_measures) or any(
                [
                    name_measure.startswith(prefix_measure)
                    for prefix_measure in self.keys_measures_prefix
                ]
            ):
                dict_metrics_aggregated[f"{self.prefix_metric}/{name_measure}"] = (
                    jnp.nanmean(dict_measures[name_measure])
                )
        return dict_metrics_aggregated

    def get_dict_metrics(self, metrics: Metric) -> Dict[str, jnp.ndarray]:
        return metrics


class AggregatorPopulationStd(Aggregator):
    """Standard deviation of the values of the measures over the population.

    f(m) = std(m)

    Can be applied to any measure, but makes more sense for state/behavior measure and
    life-aggregated metrics.
    """

    def get_initial_metrics(self) -> Metric:
        return self.get_dict_of_full_arrays(fill_value=jnp.nan, mode="scalar")

    def update_metrics(
        self,
        metrics: Metric,
        dict_measures: Dict[str, jnp.ndarray],
        are_alive: jnp.ndarray,
        are_just_dead: jnp.ndarray,
        ages: jax.Array,
    ) -> Metric:
        dict_metrics_aggregated = {}
        for name_measure in dict_measures.keys():
            if (name_measure in self.keys_measures) or any(
                [
                    name_measure.startswith(prefix_measure)
                    for prefix_measure in self.keys_measures_prefix
                ]
            ):
                dict_metrics_aggregated[f"{self.prefix_metric}/{name_measure}"] = (
                    jnp.nanstd(dict_measures[name_measure])
                )
        return dict_metrics_aggregated

    def get_dict_metrics(self, metrics: Metric) -> Dict[str, jnp.ndarray]:
        return metrics


class AggregatorPopulationMovingMean(Aggregator):
    """Mean of the values of the measures over the population, but not only on the current
    timestep, but also to previous timesteps.

    TODO: implement this aggregator

    For doing this, we use the following formula:
    if m_t is not NaN:
        f(m)_t += learning_rate * (m_t - f(m)_t)
    """

    def __init__(
        self,
        config: Dict[str, Any],
    ):
        super().__init__(config=config)
        self.learning_rate = config["learning_rate"]
        raise NotImplementedError


# ================================ Lifespan metrics ================================


class AggregatorLifespanCumulative(Aggregator):
    """Time-sum of the values of the measures over the lifespan of the agents.
    f(m)_t = sum_{i=1}^{t} m_i

    Should be applied only on immediate measures.
    """

    def get_initial_metrics(self) -> Metric:
        return self.get_dict_of_full_arrays(fill_value=jnp.nan)

    def update_metrics(
        self,
        metrics: Metric,
        dict_measures: Dict[str, jnp.ndarray],
        are_alive: jnp.ndarray,
        are_just_dead: jnp.ndarray,
        ages: jax.Array,
    ) -> Metric:

        dict_metrics_aggregated = {}
        for name_measure in dict_measures.keys():
            if (name_measure in self.keys_measures) or any(
                [
                    name_measure.startswith(prefix_measure)
                    for prefix_measure in self.keys_measures_prefix
                ]
            ):
                value_metric = metrics[f"{self.prefix_metric}/{name_measure}"]
                value_measure = dict_measures[name_measure]
                dict_metrics_aggregated[f"{self.prefix_metric}/{name_measure}"] = (
                    jnp.select(
                        condlist=[
                            ~are_alive,
                            ages == 1,
                            value_measure == jnp.nan,
                        ],
                        choicelist=[
                            jnp.nan,
                            value_measure,
                            value_metric,
                        ],
                        default=value_metric + value_measure,
                    )
                )

        return dict_metrics_aggregated

    def get_dict_metrics(self, metrics: Metric) -> Dict[str, jnp.ndarray]:
        return metrics


class AggregatorLifespanAverage(Aggregator):
    """Time-average of the values of the measures over the lifespan of the agents.
    f(m)_t = sum_{i=1}^{t} m_i / t

    Can be applied to any measure.
    """

    def get_initial_metrics(self) -> Metric:
        return self.get_dict_of_full_arrays(fill_value=jnp.nan)

    def update_metrics(
        self,
        metrics: Metric,
        dict_measures: Dict[str, jnp.ndarray],
        are_alive: jnp.ndarray,
        are_just_dead: jnp.ndarray,
        ages: jax.Array,
    ) -> Metric:

        dict_metrics_aggregated = {}
        for name_measure in dict_measures.keys():
            if (name_measure in self.keys_measures) or any(
                [
                    name_measure.startswith(prefix_measure)
                    for prefix_measure in self.keys_measures_prefix
                ]
            ):
                value_metric = metrics[f"{self.prefix_metric}/{name_measure}"]
                value_measure = dict_measures[name_measure]
                dict_metrics_aggregated[f"{self.prefix_metric}/{name_measure}"] = (
                    jnp.select(
                        condlist=[
                            ~are_alive,
                            ages == 1,
                            value_measure == jnp.nan,
                        ],
                        choicelist=[
                            jnp.nan,
                            value_measure,
                            value_metric,
                        ],
                        default=(value_metric * (ages - 1) + value_measure) / ages,
                    )
                )

        return dict_metrics_aggregated

    def get_dict_metrics(self, metrics: Metric) -> Dict[str, jnp.ndarray]:
        return metrics


class AggregatorLifespanVariation(Aggregator):
    """Difference of values of the measures over the lifespan of the agents.
    f(m)_t = m_t - m_1

    Should be applied only on behavior measures, and possibly on state measures.
    """

    def get_initial_metrics(self) -> Metric:
        return {
            "first_values": self.get_dict_of_full_arrays(fill_value=jnp.nan),
            "last_values": self.get_dict_of_full_arrays(fill_value=jnp.nan),
            "variations": self.get_dict_of_full_arrays(fill_value=jnp.nan),
            "cum_abs_variations": self.get_dict_of_full_arrays(fill_value=jnp.nan),
        }

    def update_metrics(
        self,
        metrics: Metric,
        dict_measures: Dict[str, jnp.ndarray],
        are_alive: jnp.ndarray,
        are_just_dead: jnp.ndarray,
        ages: jax.Array,
    ) -> Metric:

        dict_first_values = metrics["first_values"]
        dict_last_values = metrics["last_values"]
        dict_cum_abs_variations = metrics["cum_abs_variations"]
        new_dict_last_values = {}
        new_dict_variations = {}
        new_dict_cum_abs_variations = {}
        for name_measure in dict_measures.keys():
            if (name_measure in self.keys_measures) or any(
                [
                    name_measure.startswith(prefix_measure)
                    for prefix_measure in self.keys_measures_prefix
                ]
            ):
                value_measure = dict_measures[name_measure]
                first_value = dict_first_values.get(f"{self.prefix_metric}/{name_measure}", value_measure)
                
                # Update the first value (in case ages==1)
                dict_first_values[f"{self.prefix_metric}/{name_measure}"] = jnp.select(
                    condlist=[
                        ~are_alive,
                        ages == 1,
                        value_measure == jnp.nan,
                    ],
                    choicelist=[
                        jnp.nan,
                        value_measure,
                        first_value,
                    ],
                    default=first_value,
                )
                # Compute the new metric : value - value_last
                new_dict_variations[f"{self.prefix_metric}/{name_measure}"] = (
                    jnp.select(
                        condlist=[
                            ~are_alive,
                            ages == 1,
                            value_measure == jnp.nan,
                        ],
                        choicelist=[
                            jnp.nan,
                            0,
                            jnp.nan,
                        ],
                        default=value_measure - first_value,
                    )
                )
                # Update the last value
                new_dict_last_values[f"{self.prefix_metric}/{name_measure}"] = value_measure
                # Compute the cumulative variation
                last_value_measure = dict_last_values.get(f"{self.prefix_metric}/{name_measure}", value_measure)
                cum_abs_variation = dict_cum_abs_variations.get(f"{self.prefix_metric}/{name_measure}", 0)
                new_dict_cum_abs_variations[f"{self.prefix_metric}/{name_measure}"] = (
                    jnp.select(
                        condlist=[
                            ~are_alive,
                            ages == 1,
                            value_measure == jnp.nan,
                        ],
                        choicelist=[
                            jnp.nan,
                            0,
                            jnp.nan,
                        ],
                        default=cum_abs_variation + jnp.abs(value_measure - last_value_measure),
                    )
                )
        res = {
            "first_values": dict_first_values,
            "last_values": new_dict_last_values,
            "variations": new_dict_variations,
            "cum_abs_variations": new_dict_cum_abs_variations,
        }
        return res

    def get_dict_metrics(self, metrics: Metric) -> Dict[str, jnp.ndarray]:
        dict_metrics = metrics["variations"] # 'life_var/energy' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for key, value in metrics["cum_abs_variations"].items():
            key = key.replace("life_var", "life_var/cum_abs")
            dict_metrics[key] = value
        return dict_metrics


if __name__ == "__main__":
    config = {
        "keys_measures": ["measure1", "measure2"],
        "n_agents": 10,
    }
    agg = AggregatorLifespanCumulative(config)

    metrics = agg.get_initial_metrics()

    for t in range(10):
        dict_measures = {
            "measure1": jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            "measure2": jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        }

        metrics = agg.update_metrics(
            metrics=metrics,
            dict_measures=dict_measures,
            are_alive=jnp.array(
                [True, True, True, True, True, True, True, True, True, True]
            ),
            are_just_dead=jnp.array(
                [False, False, False, False, False, False, False, False, False, False]
            ),
            ages=jnp.array([t, t, t, t, t, t, t, t, t, t]),
        )

        print(agg.get_dict_metrics(metrics))
