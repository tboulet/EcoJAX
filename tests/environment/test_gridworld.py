from omegaconf import DictConfig, OmegaConf
import pytest

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from src.environment.gridworld import (
    AgentObservationGridworld,
    EnvStateGridworld,
    GridworldEnv,
)
from src.types_base import AgentObservation, EnvState


class TestGridworldEnv:

    @classmethod
    def setup_class(cls):
        config = OmegaConf.load("configs/env/small.yaml")
        config = OmegaConf.to_container(config, resolve=True)
        config["height"] = 10
        config["width"] = 10
        cls.n_agents_max = 10
        cls.n_agents_initial = 5
        cls.env = GridworldEnv(
            config=config,
            n_agents_max=cls.n_agents_max,
            n_agents_initial=cls.n_agents_initial,
        )

    def test_start(self):
        key_random = random.PRNGKey(1234)
        key_random, subkey = random.split(key_random)
        res = self.env.start(key_random=subkey)
        self.check_env_step_return(res)

    def test_step(self):
        key_random = random.PRNGKey(1234)
        key_random, subkey = random.split(key_random)
        state, *_ = self.env.start(key_random=subkey)
        res = self.env.step(
            key_random=subkey,
            state=state,
            actions=jnp.zeros((self.n_agents_max,)),
        )
        self.check_env_step_return(res)

    def test_step_dynamics(self):
        key_random = random.PRNGKey(1234)
        key_random, subkey = random.split(key_random)
        state, *_ = self.env.start(key_random=subkey)
        state: EnvStateGridworld = state.replace(
            positions_agents=jnp.array(
                [
                    [0, 0],
                    [0, 0],
                    [2, 2],
                    [2, 2],
                    [4, 4],
                    [5, 5],
                    [6, 6],
                    [7, 7],
                    [8, 8],
                    [9, 9],
                ]
            ),
            orientation_agents=jnp.array([0, 0, 2, 3, 0, 1, 2, 3, 0, 1]),
        )
        actions = jnp.array([1, 1, 3, 0, 1, 2, 3, 0, 1, 2])

        state, *_ = self.env.step(
            key_random=subkey,
            state=state,
            actions=actions,
        )

        assert jnp.all(
            state.positions_agents
            == jnp.array(
                [
                    [0, 9],
                    [0, 9],
                    [2, 1],
                    [2, 3],
                    [4, 3],
                    [5, 6],
                    [6, 5],
                    [7, 8],
                    [8, 7],
                    [9, 0],
                ]
            )
        ), f"Positions are wrong: {state.positions_agents}"
        assert jnp.all(
            state.orientation_agents == jnp.array([1, 1, 1, 3, 1, 3, 1, 3, 1, 3])
        ), f"Orientations are wrong: {state.orientation_agents}"
        assert jnp.all(
            state.map[:, :, self.env.dict_name_channel_to_idx["agents"]] == jnp.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            )
        ), f"Map is wrong: {state.map[:, :, self.env.dict_name_channel_to_idx['agents']]}"
    # ================ Helper methods ================

    def check_env_step_return(self, res):
        assert len(res) == 6, "The result should have 4 elements"
        (
            env_state,
            agent_observations,
            are_newborns,
            indexes_parents,
            done_env,
            info_env,
        ) = res
        assert isinstance(env_state, EnvStateGridworld)
        assert isinstance(agent_observations, AgentObservationGridworld)
        assert isinstance(are_newborns, jnp.ndarray) and are_newborns.shape == (
            self.n_agents_max,
        )
        assert (
            isinstance(indexes_parents, jnp.ndarray)
            and indexes_parents.shape[0] == self.n_agents_max
        )
