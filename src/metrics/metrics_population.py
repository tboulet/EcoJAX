from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import random


class AggregatorPopulation(ABC):
    """The base class for all aggregators of metrics over the population of agents."""

    # The types of metrics over which the aggregator can be applied
    names_metrics_type: List[str]

    @abstractmethod
    def get_aggregated_metric(
        self,
        list_metrics: jnp.ndarray, # Shape: (n_agents,)
    ) -> float:
        raise NotImplementedError


class AggregatorPopulationMean(AggregatorPopulation):

    names_metrics_type = ["immediate", "state", "lifespan"]

    def get_aggregated_metric(
        self,
        list_metrics: jnp.ndarray, # Shape: (n_agents,)
    ) -> float:
        """The function that aggregates a certain metric over the population of agents with the method 'mean'.
        It returns the mean of non-NaN values of the metric over the population of agents.
        """
        return jnp.nanmean(list_metrics)


class AggregatorPopulationStd(AggregatorPopulation):
    
    names_metrics_type = ["immediate", "state", "lifespan"]

    def get_aggregated_metric(
        self,
        list_metrics: jnp.ndarray, # Shape: (n_agents,)
    ) -> float:
        """The function that aggregates a certain metric over the population of agents with the method 'std'.
        It returns the standard deviation of non-NaN values of the metric over the population of agents.
        """
        return jnp.nanstd(list_metrics)

name_population_agg_to_AggregatorPopulationClass: Dict[str, Type[AggregatorPopulation]] = {
    "mean": AggregatorPopulationMean,
    "std": AggregatorPopulationStd,
    # "min": AggregatorPopulationMin,
    # "max": AggregatorPopulationMax,
    # "equality": AggregatorPopulationEquality,
    # "variance": AggregatorPopulationVariance,
}