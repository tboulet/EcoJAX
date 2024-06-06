from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import random


class AggregatorLifespan(ABC):
    """The base class for all aggregators of a certain measure over the lifespan of an agent.
    At age t, the metric represented by any instance of this class is an aggregation of the measures of the agent from age 0 to t.
    
    By convention, the input metrics of newborn agents should be NaN.
    
    Each class has an attribute names_metrics_type, which is a list of strings representing the types of metrics over which the aggregator can be applied.
    This is done to specify on which measures it makes sense to apply the aggregator. For example, it doesn't make sense to compute the sum of a state metric over the lifespan of an agent.
    """

    # The types of metrics over which the aggregator can be applied
    names_metrics_type: List[str]

    @abstractmethod
    def get_new_metric_value(
        self,
        current_metric_value: float,
        last_measure: float,
        new_measure: float,
        age: float,
    ) -> float:
        """Computes the new metric value at the current timestep, given the current metric value, the last measure, the new measure, and the age of the agent.

        Args:
            current_metric_value (float): the current value of the metric
            last_measure (float): the last measure of the agent
            new_measure (float): the new measure of the agent
            age (float): the age of the agent

        Returns:
            float: the new value of the metric
        """
        pass


class AggregatorLifespanCumulative(AggregatorLifespan):

    names_metrics_type = ["immediate"]

    def get_new_metric_value(
        self,
        current_metric_value: float,
        last_measure: float,
        new_measure: float,
        age: float,
    ) -> float:
        """The function that aggregates a certain measure over the lifespan of an agent with the method 'cumulative'.
        It simply sums the values of the measure over the lifespan of the agent.

        If this is the agent's first active timestep (age=1), we initialize the metric with the measure.
        Elif this agent is dead ()
        """
        return jnp.select(
            [
                age == 1,
                jnp.isnan(new_measure),
            ],
            [
                new_measure,
                jnp.nan,
            ],
            default=current_metric_value + new_measure,
        )


class AggregatorLifespanTimeAverage(AggregatorLifespan):

    names_metrics_type = ["immediate", "state"]

    def get_new_metric_value(
        self,
        current_metric_value: float,
        last_measure: float,
        new_measure: float,
        age: float,
    ) -> float:
        """The function that aggregates a certain measure over the lifespan of an agent with the method 'time_average'.
        It computes the average of the values of the measure over the lifespan of the agent.

        If the agent was just born (age=0), since the measure was computed before it was born in the step() function, we return the measure.
        If the agent was dead before this timestep (new_measure=NaN), we return NaN.
        """
        return jnp.select(
            [
                age == 0,
                jnp.isnan(new_measure),
            ],
            [
                new_measure,
                jnp.nan,
            ],
            default=(current_metric_value * age + new_measure) / (age + 1),
        )


class AggregatorLifespanVariation(AggregatorLifespan):

    names_metrics_type = ["state"]

    def get_new_metric_value(
        self,
        current_metric_value: float,
        last_measure: float,
        new_measure: float,
        age: float,
    ) -> float:
        """The function that aggregates a certain measure over the lifespan of an agent with the method 'variation'.
        It computes the variation of the values of the measure over the lifespan of the agent.
        At timestep t, the variation is v_t = m_t - m_0
        Consequently, v_0 = 0.
        We have the recursive formula v_t = v_{t-1} + (m_t - m_{t-1})

        If the agent was just born (age=0), since the measure was computed before it was born in the step() function, we return 0.
        If the agent was dead before this timestep (new_measure=NaN), we return 
        """
        return jnp.select(
            [
                age == 0,
                jnp.isnan(new_measure),
            ],
            [
                0.0,
                new_measure - current_metric_value,
            ],
            default=jnp.nan,
        )


name_lifespan_agg_to_AggregatorLifespanClass: Dict[str, Type[AggregatorLifespan]] = {
    "cum": AggregatorLifespanCumulative,
    "avg": AggregatorLifespanTimeAverage,
    "var": AggregatorLifespanVariation,
}
