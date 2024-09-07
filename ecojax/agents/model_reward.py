from abc import ABC, abstractmethod
from functools import partial
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import jax
from jax import random, tree_map
import jax.numpy as jnp
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from flax import struct
import flax.linen as nn
import optax
from flax.training import train_state
from flax.struct import PyTreeNode, dataclass


from ecojax.agents.base_agent_species import AgentSpecies
from ecojax.core.eco_info import EcoInformation
from ecojax.metrics.aggregators import Aggregator
from ecojax.models.base_model import BaseModel, FlattenAndConcatModel
from ecojax.evolution.mutator import mutate_scalar, mutation_gaussian_noise
from ecojax.models.mlp import MLP_Model
from ecojax.types import ActionAgent, ObservationAgent
import ecojax.spaces as spaces
from ecojax.utils import instantiate_class, jbreakpoint, jprint



class RewardModel(BaseModel):
    """The reward model of the agent. It maps the observation to the reward.

    Args:
        space_input (spaces.DictSpace): the input space of the model
        space_output (spaces.ContinuousSpace): the output space of the model
        func_weight (str): the method used to model the reward. Among ["constant", "linear", "hardcoded"] (see below)
        dict_reward (Optional[Dict[str, float]], optional): in case you want to hardcode the reward, the dictionnary that maps the observation components to the reward of their unit variation. Defaults to None.
    """

    space_input: spaces.DictSpace
    space_output: spaces.ContinuousSpace
    func_weight: str
    dict_reward: Optional[Dict[str, float]] = None

    def obs_to_encoding(
        self,
        x: Dict[str, jnp.ndarray],
        key_random: jnp.ndarray,
    ) -> Tuple[jnp.ndarray]:
        """The encoding of the reward model. It receives as input a dictionnary containing the following :

        - "obs" : the observation of the agent
        - "obs_next" : the next observation of the agent

        It will then extract the scalar components of the observations and compute the reward from them only.

        Args:
            x (Dict[str, jnp.ndarray]): a dictionnary containing the observations of the agent

        Returns:
            jnp.ndarray: the reward of the agent
        """
        assert isinstance(
            self.space_input, spaces.DictSpace
        ), f"Expected a TupleSpace, got {self.space_input}"
        space_obs: spaces.DictSpace = self.space_input.dict_space["obs"]
        space_next_obs: spaces.DictSpace = self.space_input.dict_space["obs_next"]
        # assert space_obs == space_next_obs, "The observation spaces must be the same"

        reward = jnp.zeros((1,))

        def init_fn_param(key: jnp.ndarray) -> jnp.ndarray:
            if self.dict_reward is not None:
                return jnp.array(self.dict_reward[name_space])
            else:
                return random.normal(key, dtype=jnp.float32)

        for name_space, space in space_obs.dict_space.items():
            if (
                isinstance(space, spaces.ContinuousSpace) and space.shape == ()
            ):  # Scalar space
                obs_component = x["obs"][name_space]
                obs_component_next = x["obs_next"][name_space]
                diff_obs = obs_component_next - obs_component

                if self.func_weight == "constant":
                    key_random, subkey = random.split(key_random)
                    alpha_diff = self.param(
                        name=f"alpha_diff_{name_space}",
                        init_fn=init_fn_param,
                        key=subkey,
                    )
                    # The additional reward due to component k will be proportional to the variation of component k : r_t += alpha_k * (o_{t+1}_k - o_t_k)
                    reward += alpha_diff * diff_obs

                elif self.func_weight == "linear":
                    # The additional reward due to component k will be proportional to the variation of component k, but the proportionality factor can vary affinely with the component value : r_t += (alpha_k + beta_k * o_t_k) * (o_{t+1}_k - o_t_k)
                    key_random, subkey = random.split(key_random)
                    alpha_diff = self.param(
                        name=f"alpha_diff_{name_space}",
                        init_fn=init_fn_param,
                        key=subkey,
                    )
                    key_random, subkey = random.split(key_random)
                    beta_diff = self.param(
                        name=f"beta_diff_{name_space}",
                        init_fn=init_fn_param,
                        key=subkey,
                    )
                    reward += (alpha_diff + obs_component * beta_diff) * diff_obs

                elif self.func_weight == "hardcoded":
                    assert (
                        self.dict_reward is not None
                    ), "On hardcoded reward mode, the dict_reward dictionnary must be provided in the reward model config (agents.reward_model.dict_reward)"
                    reward += diff_obs * self.dict_reward[name_space]

                elif self.func_weight == "one":
                    reward += 1

                else:
                    raise NotImplementedError(
                        f"Function {self.func_weight} not implemented"
                    )

        return reward
