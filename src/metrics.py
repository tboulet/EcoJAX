from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import random


class AggregatorLifespan(ABC):
    """The base class for all aggregators of a certain measure over the lifespan of an agent."""

    # The types of measure over which the aggregator can be applied
    names_measure_type: List[str]

    @abstractmethod
    def get_new_metric_value(
        self,
        current_metric_value: float,
        measure: float,
        age: float,
    ) -> float:
        pass


class AggregatorLifespanCumulative(AggregatorLifespan):

    names_measure_type = ["cumulative"]

    def get_new_metric_value(
        self,
        current_metric_value: float,
        measure: float,
        age: float,
    ) -> float:
        """The function that aggregates a certain measure over the lifespan of an agent with the method 'cumulative'.
        It simply sums the values of the measure over the lifespan of the agent.

        If the agent was just born (age=0), since the measure was computed before it was born in the step() function, we return 0.
        If the agent was dead before this timestep (measure=NaN), we return NaN.
        """
        return jnp.select(
            [
                age == 0,
                jnp.isnan(measure),
            ],
            [
                0.0,
                jnp.nan,
            ],
            default=current_metric_value + measure,
        )


name_lifespan_agg_to_AggregatorClass: Dict[str, Type[AggregatorLifespan]] = {
    "cumulative": AggregatorLifespanCumulative,
}
