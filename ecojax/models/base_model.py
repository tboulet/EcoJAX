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
        observation_class (Type[ObservationAgent]): the JAX class of the observation agent
        n_actions (int): the number of possible actions of the agent
        return_modes (List[str]): the list of the possible return modes of the model. Can contain (and at least 1 of) the following strings:
            1) "probs" : the model returns the probabilities of the actions.
            2) "logits" : the model returns the logits of the actions. Incompatible with "probs".
            3) "q_values" : the model returns the Q-values of the actions.
            4) "state_value" : the model returns the state value, as a scalar.
    """

    observation_space_dict: Dict[str, EcojaxSpace]
    observation_class: Type[ObservationAgent]
    n_actions: int
    return_modes: List[str]

    def obs_to_encoding(
        self, obs: ObservationAgent, key_random: jnp.ndarray
    ) -> jnp.ndarray:
        """Converts the observation to a vector encoding that can be processed by the model."""
        raise NotImplementedError(
            "The method obs_to_encoding must be implemented in the subclass."
        )

    def sample_observation(self, key_random: jnp.ndarray) -> ObservationAgent:
        # Sample the observation from the different spaces
        kwargs_obs: Dict[str, np.ndarray] = {}
        for key_dict, space in self.observation_space_dict.items():
            key_random, subkey = random.split(key_random)
            kwargs_obs[key_dict] = space.sample(key_random=subkey)
        return self.observation_class(**kwargs_obs)
        
        
    def get_initialized_variables(
        self, key_random: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        """Initializes the model's variables and returns them as a dictionary.
        This is a wrapper around the init method of nn.Module, which creates an observation for initializing the model.
        """
        # Sample the observation from the different spaces
        key_random, subkey = random.split(key_random)
        obs = self.sample_observation(subkey)

        # Run the forward pass to initialize the model
        key_random, subkey = random.split(key_random)
        return nn.Module.init(
            self,
            key_random,
            obs=obs,
            key_random=subkey,
        )

    def get_action_and_prob(
        self, x: jnp.ndarray, key_random: jnp.ndarray
    ) -> Tuple[ActionAgent, jnp.ndarray]:
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

    @nn.compact
    def __call__(
        self, obs: ObservationAgent, key_random: jnp.ndarray,
    ) -> Tuple[jnp.ndarray]:
        """The forward pass of the model. It maps the observation to the output in the right format.
        The observation input is given as an ObservationAgent object, ie a JAX dataclass of component in the corresponding observation space.
        The output can be returned in three different modes and with or without the probabilit(y/ies) of the action(s).

        Args:
            obs (ObservationAgent): the observation of the agent
            key_random (jnp.ndarray): the random key used for any random operation in the forward pass
            
        Returns:
            Tuple[jnp.ndarray]: a tuple of the requested outputs
        """

        # Convert the observation to a vector encoding
        x = self.obs_to_encoding(obs, key_random)

        # Return the output as a tuple of the requested modes
        list_outputs : List[jnp.ndarray] = []
        for mode in self.return_modes:
            if mode == "probs":
                probs = nn.softmax(nn.Dense(features=self.n_actions)(x))
                list_outputs.append(probs)
            elif mode == "logits":
                assert "probs" not in self.return_modes, "Cannot return both logits and probs"
                logits = nn.Dense(features=self.n_actions)(x)
                list_outputs.append(logits)
            elif mode == "q_values":
                q_values = nn.Dense(features=self.n_actions)(x)
                list_outputs.append(q_values)
            elif mode == "state_value":
                state_value = nn.Dense(features=1)(x)
                list_outputs.append(state_value)
            else:
                raise ValueError(f"Unknown return mode: {mode}")
        return tuple(list_outputs)

    def get_table_summary(self) -> Dict[str, Any]:
        """Returns a table that summarizes the model's parameters and shapes."""

        kwargs_obs: Dict[str, np.ndarray] = {}
        key_random = jax.random.PRNGKey(0)
        for key_dict, space in self.observation_space_dict.items():
            kwargs_obs[key_dict] = space.sample(key_random=key_random)
        obs = self.observation_class(**kwargs_obs)
        return f"Model summary: {nn.tabulate(self, rngs=key_random)(obs, key_random)}"
