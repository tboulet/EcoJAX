from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Type, Union, Any
import numpy as np
import jax.numpy as jnp

from ecojax.core.eco_info import EcoInformation
from ecojax.spaces import EcojaxSpace
from ecojax.types import ActionAgent, StateEnv, ObservationAgent


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
        self.config = config
        self.n_agents_max = n_agents_max
        self.n_agents_initial = n_agents_initial

    @abstractmethod
    def reset(
        self,
        key_random: jnp.ndarray,
    ) -> Tuple[
        StateEnv,
        ObservationAgent,
        EcoInformation,
        bool,
        Dict[str, Any],
    ]:
        """Start the environment. This initialize the state and also returns the initial observations and eco variables of the agents.

        Args:
            key_random (jnp.ndarray): the random key used for the initialization

        Returns:
            observations_agents (ObservationAgentGridworld): the new observations of the agents, of attributes of shape (n_max_agents, dim_observation_components)
            dict_reproduction (Dict[int, List[int]]): a dictionary indicating the indexes of the parents of each newborn agent. The keys are the indexes of the newborn agents, and the values are the indexes of the parents of the newborn agents.
            done (bool): whether the environment is done
            info (Dict[str, Any]): the info of the environment
        """
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        state: StateEnv,
        actions: ActionAgent,
        key_random: jnp.ndarray,
    ) -> Tuple[
        StateEnv,
        ObservationAgent,
        EcoInformation,
        bool,
    ]:
        """Perform one step of the Gridworld environment.

        Args:
            key_random (jnp.ndarray): the random key used for this step
            jnp.ndarray: the observations to give to the agents, of shape (n_max_agents, dim_observation)
            actions (ActionAgent): the actions to perform

        Returns:
            observations_agents (ObservationAgent): the new observations of the agents, of attributes of shape (n_max_agents, dim_observation_components)
            eco_information (EcoInformation): the ecological information of the environment regarding what happened at t. It should contain the following:
                1) are_newborns_agents (jnp.ndarray): a boolean array indicating which agents are newborns at this step
                2) indexes_parents_agents (jnp.ndarray): an array indicating the indexes of the parents of the newborns at this step
                3) are_dead_agents (jnp.ndarray): a boolean array indicating which agents are dead at this step (i.e. they were alive at t but not at t+1)
                    Note that an agent index could see its are_dead_agents value be False while its are_newborns_agents value is True, if the agent die and another agent is born at the same index
            done (bool): whether the environment is done
        """
        raise NotImplementedError


    @abstractmethod
    def get_observation_space_dict(self) -> Dict[str, EcojaxSpace]:
        """Return a dictionnary describing the observation space of the environment.

        The keys of the dictionnary are the names of the observation components.

        The values are the shapes of the observation components.
        If a value is an integer n, it means the observation component will be an integer between 0 and n-1.
        If a value is a tuple (n1, n2, ...), it means the observation component will be an array of shape (n1, n2, ...).

        Each agent will expect its observation to be an ObservationAgent object that contains each key as attribute, with the corresponding shape.

        Returns:
            Dict[str, Space]: the observation space dictionnary
        """
        raise NotImplementedError

    @abstractmethod
    def get_action_space_dict(self) -> Dict[str, EcojaxSpace]:
        """Return a dictionnary describing the action space of the environment.

        The keys of the dictionnary are the names of the action components.

        The values are the shapes of the action components.
        If a value is an integer n, it means the action component will be an integer between 0 and n-1.
        If a value is a tuple (n1, n2, ...), it means the action component will be an array of shape (n1, n2, ...).

        The agents will be expected to send an ActionAgent object that contains each key as attribute, with the corresponding shape.

        Returns:
            Dict[str, Space]: the action space dictionnary
        """
        raise NotImplementedError

    @abstractmethod
    def get_class_observation_agent(self) -> Type[ObservationAgent]:
        """Return the class of the observation of the agents.

        Returns:
            Type[ObservationAgent]: the class of the observation of the agents
        """
        raise NotImplementedError

    @abstractmethod
    def get_class_action_agent(self) -> Type[ActionAgent]:
        """Return the class of the action of the agents.

        Returns:
            Type[ActionAgent]: the class of the action of the agents
        """
        raise NotImplementedError

    @abstractmethod
    def render(self, state : StateEnv) -> None:
        """Do the rendering of the environment. This can be a visual rendering or a logging of the state of any kind."""
        return
