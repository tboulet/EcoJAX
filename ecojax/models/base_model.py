from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Type

import numpy as np

import flax.linen as nn
from jax import random
import jax.numpy as jnp

from ecojax.spaces import Space
from ecojax.types import ActionAgent, ObservationAgent


class BaseModel(nn.Module, ABC):
    """The base class for all models. A model is a way to map observations and weights to actions.
    This abstract class subclasses nn.Module, which is the base class for all Flax models.
    """

    config: Dict[str, Any]
    observation_space_dict: Dict[str, Space]
    action_space_dict: Dict[str, Space]
    observation_class: Type[ObservationAgent]
    action_class: Type[ActionAgent]

    def get_initialized_variables(
        self, key_random: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        """Initializes the model's variables and returns them as a dictionary.
        This is a wrapper around the init method of nn.Module, which creates an observation for initializing the model.
        """
        # Sample the observation from the different spaces
        kwargs_obs: Dict[str, np.ndarray] = {}
        for key_dict, space in self.observation_space_dict.items():
            key_random, subkey = random.split(key_random)
            kwargs_obs[key_dict] = space.sample(key_random=subkey)
        obs = self.observation_class(**kwargs_obs)

        # Run the forward pass to initialize the model
        key_random, key_random2 = random.split(key_random)
        return nn.Module.init(
            self,
            key_random,
            obs=obs,
            key_random=key_random2,
        )

    @abstractmethod
    def __call__(
        self, obs: ObservationAgent, key_random: jnp.ndarray
    ) -> Tuple[ActionAgent, jnp.ndarray]:
        """The forward pass of the model. Maps observations to actions. Also returns the probability of the sampled action
        since it is useful for some algorithms.

        Args:
            obs (ObservationAgent): the observation of the agent
            key_random (jnp.ndarray): the random key used for any random operation in the forward pass

        Raises:
            NotImplementedError: _description_

        Returns:
            action (ActionAgent): the action of the agent
            prob_action_sampled (jnp.ndarray): the probability of the sampled action, as a scalar
        """
        raise NotImplementedError
