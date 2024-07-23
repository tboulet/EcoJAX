from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Type, Union, Any
import numpy as np
import jax.numpy as jnp

from ecojax.core.eco_info import EcoInformation
from ecojax.spaces import EcojaxSpace
from ecojax.types import ActionAgent, StateEnv, ObservationAgent, StateSpecies


class EcoEnvironment(ABC):
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
        self.agent_species = None
        
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
            state (StateEnv): the initial state of the environment
            observations_agents (ObservationAgentGridworld): the new observations of the agents, of attributes of shape (n_max_agents, dim_observation_components)
            eco_information (EcoInformation): the ecological information of the environment regarding what happened at t. It should contain the following:
            done (bool): whether the environment is done
            info (Dict[str, Any]): the info of the environment
        """
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        state: StateEnv,
        actions: ActionAgent, # Batched
        key_random: jnp.ndarray,
        state_species: Optional[StateSpecies] = None,
    ) -> Tuple[
        StateEnv,
        ObservationAgent,
        EcoInformation,
        bool,
        Dict[str, Any],
    ]:
        """Perform one step of the Gridworld environment.

        Args:
            state (StateEnv): the state of the environment at t
            actions (ActionAgent): the actions of the agents at t, of attributes of shape (n_max_agents, dim_action_components)
            key_random (jnp.ndarray): the random key used for the step
            state_species (StateSpecies): the state of the species of agents at t (optional)

        Returns:
            state_new (StateEnv): the new state of the environment at t+1
            observations_agents (ObservationAgent): the new observations of the agents, of attributes of shape (n_max_agents, dim_observation_components)
            eco_information (EcoInformation): the ecological information of the environment regarding what happened at t. It should contain the following:
                1) are_newborns_agents (jnp.ndarray): a boolean array indicating which agents are newborns at this step
                2) indexes_parents_agents (jnp.ndarray): an array indicating the indexes of the parents of the newborns at this step
                3) are_dead_agents (jnp.ndarray): a boolean array indicating which agents are dead at this step (i.e. they were alive at t but not at t+1)
                    Note that an agent index could see its are_dead_agents value be False while its are_newborns_agents value is True, if the agent die and another agent is born at the same index
            done (bool): whether the environment is done
            info (Dict[str, Any]): the info of the environment
        """
        raise NotImplementedError

    @abstractmethod
    def get_observation_space(self) -> EcojaxSpace:
        """Return the observation space of the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def get_action_space(self) -> EcojaxSpace:
        """Return the action space of the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def render(self, state: StateEnv, force_render : bool = False) -> None:
        """Do the rendering of the environment. This can be a visual rendering or a logging of the state of any kind.
        
        Args:
            state (StateEnv): the state of the environment to render
            force_render (bool): whether to force the rendering even if the environment is not in a state where it should be rendered
        """
        return

    def compute_on_render_behavior_measures(
        self,
        react_fn: Callable[
            [
                StateSpecies,
                ObservationAgent,
                EcoInformation,
                jnp.ndarray,
            ],
            Tuple[
                StateSpecies,
                ActionAgent,
                Dict[str, jnp.ndarray],
            ],
        ],
        key_random: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray]:
        """Perform a battery of tests on the agents using the given act_fn and artificial observations.

        Args:
            act_fn (Callable[[jnp.ndarray, ObservationAgent], ActionAgent]): the function that maps the observation(s) to the action(s)
            key_random (jnp.ndarray): the random key used for the tests

        Returns:
            Dict[str, jnp.ndarray]: a dictionary containing the results of the tests
        """
        return {}
