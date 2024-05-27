from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Any
import numpy as np
import jax.numpy as jnp

from src.types_base import StateEnv, ObservationAgent


class BaseEcoEnvironment(ABC):
    """The base class for any EcoJAX environment.
    An Eco-environment is a simulation with which a (non constant) set of agents will interact and evolve.
    It contains those 2 general notions :
    - Agent-wise input/output dynamics : how the environment reacts to the actions of each agents and what observation it returns to each of them ?
    - Evolution dynamics : when should one (or more) parents reproduce ?

    For obeying the constant shape paradigm (even if the number of agents in the simulation is not constant),
    the environment should always return a JAX-batched observation,
    and receive a JAX-batched action.
    This imply that any BaseEnvironment instance should maintain, for information concerning agents, (n_agents_max, **dim_agent_info) components, and maintain
    a population of existing agents and "ghost" agents (agents that are not existing in the simulation at a given time).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        n_agents_max: int,
        n_agents_initial: int,
    ):
        """The constructor of the BaseEnvironment class.

        Args:
            config (Dict[str, Any]): the environment configuration
            n_agents_max (int): the maximal number of agents allowed to exist in the simulation.
            n_agents_initial (int): the initial number of agents in the simulation
        """

    @abstractmethod
    def reset(
        self,
        key_random: jnp.ndarray,
    ) -> Tuple[
        StateEnv,
        ObservationAgent,
        Dict[int, List[int]],
        bool,
        Dict[str, Any],
    ]:
        """Start the environment. This initialize the state and also returns the initial observations and eco variables of the agents.

        Args:
            key_random (jnp.ndarray): the random key used for the initialization

        Returns:
            state (StateEnvGridworld): the initial state of the environment
            observations_agents (ObservationAgentGridworld): the new observations of the agents, of attributes of shape (n_max_agents, dim_observation_components)
            dict_reproduction (Dict[int, List[int]]): a dictionary indicating the indexes of the parents of each newborn agent. The keys are the indexes of the newborn agents, and the values are the indexes of the parents of the newborn agents.
            done (bool): whether the environment is done
            info (Dict[str, Any]): the info of the environment
        """

    @abstractmethod
    def step(
        self,
        key_random: jnp.ndarray,
        state: StateEnv,
        actions: jnp.ndarray,
    ) -> Tuple[
        StateEnv,
        ObservationAgent,
        Dict[int, List[int]],
        bool,
        Dict[str, Any],
    ]:
        """Perform one step of the Gridworld environment.

        Args:
            key_random (jnp.ndarray): the random key used for this step
            jnp.ndarray: the observations to give to the agents, of shape (n_max_agents, dim_observation)
            state (StateEnvGridworld): the state of the environment
            actions (jnp.ndarray): the actions to perform

        Returns:
            state (StateEnvGridworld): the new state of the environment
            observations_agents (ObservationAgentGridworld): the new observations of the agents, of attributes of shape (n_max_agents, dim_observation_components)
            dict_reproduction (Dict[int, List[int]]): a dictionary indicating the indexes of the parents of each newborn agent. The keys are the indexes of the newborn agents, and the values are the indexes of the parents of the newborn agents.
            done (bool): whether the environment is done
            info (Dict[str, Any]): the info of the environment
        """
        raise NotImplementedError
        # Apply the actions of the agents on the environment
        state = f(state, actions)
        # Get which agents are newborns and which are their parents
        dict_reproduction = f(state)
        # Extract the observations of the agents
        observations_agents = f(state)
        # Return the results
        return (
            state,
            observations_agents,
            dict_reproduction,
            done,
            info,
        )

    @abstractmethod
    def render(self, state: StateEnv) -> None:
        """Do the rendering of the environment. This can be a visual rendering or a logging of the state of any kind.

        Args:
            state (StateEnv): the state of the environment
        """
        return
