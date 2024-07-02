from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
import os
from time import sleep
from typing import Any, Dict, List, Tuple, Type, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.scipy.signal import convolve2d
from flax.struct import PyTreeNode, dataclass
from jax.debug import breakpoint as jbreakpoint

from ecojax.core.eco_info import EcoInformation
from ecojax.environment import EcoEnvironment
from ecojax.spaces import EcojaxSpace, Discrete, Continuous
from ecojax.types import ActionAgent, ObservationAgent, StateEnv


class Aggregator(ABC):

    def __init__(self, config: Dict[str, Any]):
        self.config: Dict[str, Any] = config
        self.keys_measures: List[str] = config["keys_measures"]
        self.keys_measures_prefix: List[str] = config.get("keys_measures_prefix", [])
        self.n_agents: str = config["n_agents"]
        self.prefix_metric: str = config.get("prefix_metric", None)
        self.log_final: bool = config.get("log_final", False)

    @abstractmethod
    def get_initial_metrics(self) -> PyTreeNode:
        """Return the initial metrics of the aggregator."""
        raise NotImplementedError
    
    @abstractmethod
    def update_metrics(
        self,
        metrics : PyTreeNode,
        dict_measures : Dict[str, jnp.ndarray],
        are_alive: jnp.ndarray,
        are_just_dead: jnp.ndarray,
        ages: jax.Array,
    ) -> PyTreeNode:
        """Return the updated metrics of the aggregator."""
        raise NotImplementedError

    @abstractmethod
    def get_dict_metrics(self, metrics : PyTreeNode) -> Dict[str, jnp.ndarray]:
        """Return the metrics of the aggregator as a dictionnary."""
        return {}






@dataclass
class LifespanCumulative(PyTreeNode):
    dict_cum_values: Dict[str, jnp.ndarray]


class AggregatorLifespanCumulative(Aggregator):

    def get_initial_metrics(self) -> LifespanCumulative:
        return LifespanCumulative(
            dict_cum_values={
                f"{self.prefix_metric}/{name_measure}": jnp.full(
                    (self.n_agents,), jnp.nan
                )
                for name_measure in self.keys_measures
            }
        )

    def update_metrics(
        self,
        metrics : LifespanCumulative,
        dict_measures : Dict[str, jnp.ndarray],
        are_alive: jnp.ndarray,
        are_just_dead: jnp.ndarray,
        ages: jax.Array,
    ) -> LifespanCumulative:

        def get_new_value_metric(
            value_metric : jnp.ndarray,
            value_measure : jnp.ndarray,
        ):
            
            return jnp.select(
                condlist=[
                    ~are_alive,
                    ages == 1,
                ],
                choicelist=[
                    jnp.nan,
                    value_measure,
                ],
                default=value_metric + value_measure,
            )

        dict_metrics_aggregated = {}
        for name_measure, value_measure in dict_measures.items():
            if (name_measure in self.keys_measures) or any(
                [
                    name_measure.startswith(prefix_measure)
                    for prefix_measure in self.keys_measures_prefix
                ]
            ):
                value_metric = metrics.dict_cum_values[f"{self.prefix_metric}/{name_measure}"]
                new_value_metric = get_new_value_metric(value_metric, value_measure)
                dict_metrics_aggregated[f"{self.prefix_metric}/{name_measure}"] = new_value_metric
                
        return metrics.replace(dict_cum_values=dict_metrics_aggregated)
        
    def get_dict_metrics(self, metrics: LifespanCumulative) -> Dict[str, jnp.ndarray]:
        return metrics.dict_cum_values


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

