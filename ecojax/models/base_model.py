from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Type

import jax
import numpy as np

import flax.linen as nn
from jax import random
import jax.numpy as jnp

from ecojax.spaces import Continuous, Discrete, EcojaxSpace
from ecojax.types import ActionAgent, ObservationAgent


class BaseModel(nn.Module, ABC):
    """The base class for all models. A model is a way to map observations to actions.
    This abstract class subclasses nn.Module, which is the base class for all Flax models.
    
    For subclassing this class, users need to add the dataclass parameters and implement the __call__ method.
    
    Args:
        observation_space_dict (Dict[str, EcojaxSpace]): a dictionary of the observation spaces. The keys are the names of the observation components, and the values are the corresponding spaces.
        action_space_dict (Dict[str, EcojaxSpace]): a dictionary of the action spaces. The keys are the names of the action components, and the values are the corresponding spaces.
        observation_class (Type[ObservationAgent]): the JAX class of the observation agent
        action_class (Type[ActionAgent]): the JAX class of the action agent
    """

    observation_space_dict: Dict[str, EcojaxSpace]
    action_space_dict: Dict[str, EcojaxSpace]
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

    def get_action_and_prob(self, x : jnp.ndarray, key_random: jnp.ndarray) -> Tuple[ActionAgent, jnp.ndarray]:
        """Computes the action and the probability of taking that action given the input x.
        It use the knowledge of the action class and the action space dictionary to generate the action in the right format.

        Args:
            x (jnp.ndarray): a vector of a certain (n,) shape
            key_random (jnp.ndarray): the random key used for any random operation in the forward pass

        Returns:
            action (ActionAgent): the action of the agent
            prob_action_sampled (jnp.ndarray): the probability of the sampled action, as a scalar
        """
        # Generate the outputs for each action space
        kwargs_action = {}
        prob_action_sampled = 1.0
        for name_action_component, space in self.action_space_dict.items():
            key_random, subkey = random.split(key_random)
            if isinstance(space, Discrete):
                action_component_logits = nn.Dense(features=space.n)(x)
                action_component_probs = nn.softmax(action_component_logits)
                action_component_sampled = jax.random.categorical(
                    subkey, action_component_logits
                )
                kwargs_action[name_action_component] = action_component_sampled
                prob_action_sampled *= action_component_probs[action_component_sampled]
            elif isinstance(space, Continuous):
                mean = nn.Dense(features=np.prod(space.shape))(x)
                log_std = nn.Dense(features=np.prod(space.shape))(x)
                std = jnp.exp(log_std)
                action_component_sampled = mean + std * random.normal(
                    subkey, shape=mean.shape
                )
                kwargs_action[name_action_component] = action_component_sampled
                # Assuming a standard normal distribution for the purpose of the probability
                action_component_prob = (
                    1.0 / jnp.sqrt(2.0 * jnp.pi * std**2)
                ) * jnp.exp(-0.5 * ((action_component_sampled - mean) / std) ** 2)
                prob_action_sampled *= jnp.prod(action_component_prob)
            else:
                raise ValueError(f"Unknown space type for action: {type(space)}")

        # Return the action
        return self.action_class(**kwargs_action), prob_action_sampled
    
    
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
