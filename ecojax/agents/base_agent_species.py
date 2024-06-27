from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, List, Tuple, Type, Union

import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from flax import struct
import flax.linen as nn

from ecojax.core.eco_info import EcoInformation
from ecojax.models.base_model import BaseModel
from ecojax.spaces import EcojaxSpace
from ecojax.types import ObservationAgent, ActionAgent, StateSpecies


class BaseAgentSpecies(ABC):
    """The base class for any species of agents.
    A species of agents represents a general method for eco-evolutionary algorithms environment-agnostic.

    It contains those two general notions :
    - How an agent reacts to its observation "o", where "o" is the observation received by the agent from the environment ?
    - What is the reproduction mecanism between agents indexes "i1", "i2", ..., "in", where those indexes are received from the environment ?

    For obeying the constant shape paradigm, even if the number of agents in the simulation is not constant, the number of "potential agents" is constant and equal to n_agents_max.
    This imply that any BaseAgentSpecies instance should maintain a population of n_agents_max agents, even if some of them are not existing in the simulation at a given time.
    """

    def __init__(
        self,
        config: Dict,
        n_agents_max: int,
        n_agents_initial: int,
        model: nn.Module,
    ):
        """The constructor of the BaseAgentSpecies class. It initializes the species of agents with the configuration.
        Elements allowing the interactions with the environment are also given as input : the numbers of agents, and the observation and action spaces.

        The observation and action space dictionnary are objects allowing to describe the observation and action spaces of the agents. More information can be found in the documentation of the environment at the corresponding methods.

        Args:
            config (Dict): the agent species configuration
            n_agents_max (int): the maximal number of agents allowed to exist in the simulation. This will also be the number of agents that are simulated every step (even if not all agents exist in the simulation at a given time)
            n_agents_initial (int): the initial number of agents in the simulation
            model (nn.Module): the model used by the agents to react to their observations
        """
        self.config = config
        self.n_agents_max = n_agents_max
        self.n_agents_initial = n_agents_initial
        self.model: BaseModel = model

    @abstractmethod
    def reset(self, key_random: jnp.ndarray) -> StateSpecies:
        """Initialize the agents of the species. This should in particular initialize the agents species' state,
        i.e. the JAX dataclass that will contain the varying information of the agents species.

        Args:
            key_random (jnp.ndarray): the random key, of shape (2,)
        """

    @abstractmethod
    def react(
        self,
        state: StateSpecies,
        batch_observations: ObservationAgent,
        eco_information: EcoInformation,
        key_random: jnp.ndarray,
    ) -> ActionAgent:
        """A function through which the agents reach to their observations and return their actions.
        It also handles the reproduction of the agents if required by the environment.

        Args:
            state (StateSpecies): the state of the species, as a StateSpecies object.
            batch_observations (jnp.ndarray): the observations, as a JAX structure of components, each of shape (n_agents_max, **dim_obs_component).
                It is composed of n_agents_max observations, each of them corresponding to the observation that the i-th indexed agent would receive.
            eco_information (EcoInformation): the eco-information of the environment, as an EcoInformation object.
            key_random (jnp.ndarray): the random key, of shape (2,)

        Returns:
            action (ActionAgent): the actions of the agents, as a ActionAgent object of components of shape (n_agents_max, **dim_action_component).
        """
        raise NotImplementedError
