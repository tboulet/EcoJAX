from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, List, Tuple, Type, Union

import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from flax import struct
import flax.linen as nn

from src.spaces import Space
from src.types_base import ObservationAgent, ActionAgent


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
        self.model = model

    @abstractmethod
    def init(self, key_random: jnp.ndarray) -> None:
        """Initialize the agents of the species. This should in particular initialize the agents species' state,
        i.e. the JAX dataclass that will contain the varying information of the agents species.

        Args:
            key_random (jnp.ndarray): the random key, of shape (2,)
        """

    @abstractmethod
    def react(
        self,
        key_random: jnp.ndarray,
        batch_observations: ObservationAgent,
        dict_reproduction: Dict[int, List[int]],
    ) -> ActionAgent:
        """A function through which the agents reach to their observations and return their actions.
        It also handles the reproduction of the agents if required by the environment.

        Args:
            key_random (jnp.ndarray): the random key, of shape (2,)
            batch_observations (jnp.ndarray): the observations, as a JAX structure of components, each of shape (n_agents_max, **dim_obs_component).
                It is composed of n_agents_max observations, each of them corresponding to the observation that the i-th indexed agent would receive.
            dict_reproduction (Dict[int, List[int]]): a dictionary indicating the indexes of the parents of each newborn agent. The keys are the indexes of the newborn agents, and the values are the indexes of the parents of the newborn agents.

        Returns:
            action (ActionAgent): the actions of the agents, as a ActionAgent object of components of shape (n_agents_max, **dim_action_component).
        """
        raise NotImplementedError
        # Process with the reproduction here...
        ...
        # React agent-by-agent to the observations
        actions = ...
        # Return the actions
        return actions
