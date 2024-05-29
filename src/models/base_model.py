from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type

import numpy as np

import flax.linen as nn

from src.spaces import Space
from src.types_base import ActionAgent, ObservationAgent


class BaseModel(nn.Module, ABC):
    """The base class for all models. A model is a way to map observations and weights to actions.
    This abstract class subclasses nn.Module, which is the base class for all Flax models.
    """

    config: Dict[str, Any]
    observation_space_dict: Dict[str, Space]
    action_space_dict: Dict[str, Space]
    observation_class: Type[ObservationAgent]
    action_class: Type[ActionAgent]

    @abstractmethod
    def get_initialized_variables(self, key_random: np.ndarray) -> None:
        """Initializes the model with the given configuration.

        Args:
            key_random (np.ndarray): the random key used for initialization

        Returns:
            None
        """
        pass

    @abstractmethod
    def __call__(self, obs: ObservationAgent) -> ActionAgent:
        pass
