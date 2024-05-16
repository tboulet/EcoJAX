from abc import ABC, abstractmethod
from typing import Union, Any
import numpy as np
import jax.numpy as jnp


class BaseEnvironment(ABC):
    """The base class for any EcoJAX environment.
    
    Steps in the development of this object :
    1) food dynamics : the env should be able to modelize creation of food; to form myself to the JAX library
    2) answer to agents actions : the env should be able to move the agents when receiving their actions
    3) eating : the env should be able to add the eating mechanism
    4) observation : the env should be able to return the observation of the agents
    5) reproduction : the env should be able to add the reproduction mechanism 
    """
    
    def __init__(self, config) -> None:
        self.config = config
    
    @abstractmethod
    def start(self) -> Any:
        """Start the environment simulation.
        """

    @abstractmethod
    def step(self, actions : jnp.ndarray = None) -> Any:
        """Step the environment by one timestep.
        """
    
    @abstractmethod
    def get_RGB_map(self) -> Any:
        """Return the RGB map of the environment.
        """