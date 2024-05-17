# Gridworld EcoJAX environment

from functools import partial
from typing import Any
import jax.numpy as jnp
import jax
from jax.scipy.signal import convolve2d

from src.environment.base_env import BaseEnvironment
from src.utils import sigmoid, logit


class GridworldEnv(BaseEnvironment):
    """The Gridworld environment."""

    def __init__(self, config) -> None:
        super().__init__(config)
        # Environment Parameters
        self.timestep: int = 0
        self.width: int = config["width"]
        self.height: int = config["height"]
        self.type_border: str = config["type_border"]
        # Sun Parameters
        self.period_sun: int = config["period_sun"]
        self.method_sun: int = config["method_sun"]
        self.radius_sun_effect: int = config["radius_sun_effect"]
        self.radius_sun_perception: int = config["radius_sun_perception"]
        # Food Dynamics
        self.p_initial_food: float = config["p_initial_food"]
        self.logit_base_grow_food: float = logit(config["p_base_grow_food"])
        self.logit_base_die_food: float = logit(config["p_base_die_food"])
        self.factor_sun_grow_food: float = config["factor_sun_grow_food"]
        self.factor_food_grow_food: float = config["factor_food_grow_food"]
        self.radius_food_grow_food: int = config["radius_food_grow_food"]
        self.kernel_food_grow_food: jnp.ndarray = jnp.ones(
            (self.radius_food_grow_food, self.radius_food_grow_food)
        ) / (self.radius_food_grow_food**2)
        self.factor_food_kill_food: float = config["factor_food_kill_food"]
        self.radius_food_kill_food: int = config["radius_food_kill_food"]
        self.kernel_food_kill_food: jnp.ndarray = jnp.ones(
            (self.radius_food_kill_food, self.radius_food_kill_food)
        ) / (self.radius_food_kill_food**2)

    def reset(self, seed: int = None) -> None:
        self.key_random = jax.random.PRNGKey(seed)
        # Initialize the map
        self.map = jnp.zeros(shape=(self.width, self.height, 3))
        # Initialize the food
        self.key_random, subkey = jax.random.split(self.key_random)
        self.map = self.map.at[:, :, 1].set(
            jax.random.bernoulli(
                key=subkey,
                p=self.p_initial_food,
                shape=(self.width, self.height),
            )
        )
        # Initialize the sun
        self.latitude_sun = self.height // 2

    def step(self, actions: jnp.ndarray = None):
        # Update the sun
        self.update_sun()
        map_is_sunny = jnp.zeros((self.width, self.height))
        map_is_sunny = map_is_sunny.at[
            self.latitude_sun
            - self.radius_sun_effect : self.latitude_sun
            + self.radius_sun_effect,
            :,
        ].set(1.0)
        # Grow food
        map_food = self.map[:, :, 1]  # (W, H)
        map_n_food_in_radius_food_grow_food = convolve2d(
            map_food,
            self.kernel_food_grow_food,
            mode="same",
        )
        map_n_food_in_radius_food_kill_food = convolve2d(
            map_food,
            self.kernel_food_kill_food,
            mode="same",
        )
        map_food_probs = sigmoid(
            x=self.logit_base_grow_food * (1 - map_food)
            + self.factor_sun_grow_food * map_is_sunny
            + self.factor_food_grow_food * map_n_food_in_radius_food_grow_food
            - self.factor_food_kill_food * map_n_food_in_radius_food_kill_food
            - self.logit_base_die_food * map_food
        )
        self.key_random, subkey = jax.random.split(self.key_random)
        map_food = jax.random.bernoulli(
            key=subkey,
            p=map_food_probs,
            shape=(self.width, self.height),
        )
        # Update the map
        self.map = self.map.at[:, :, 1].set(map_food)
        # Update the timestep
        self.timestep += 1

    def get_RGB_map(self) -> Any:
        return self.map

    # Helper functions

    def update_sun(self):
        # Remove the red coloration of the sun
        self.map = self.map.at[self.latitude_sun, :, 0].set(0.0)
        # Update the latitude of the sun depending on the method
        if self.method_sun == "none":
            self.factor_sun_grow_food = 0.0  # this prevents the sun effect
        elif self.method_sun == "fixed":
            pass
        elif self.method_sun == "random":
            self.key_random, subkey = jax.random.split(self.key_random)
            self.latitude_sun = jax.random.randint(
                key=subkey,
                shape=(),
                minval=0,
                maxval=self.height,
            )
        elif self.method_sun == "brownian":
            self.key_random, subkey = jax.random.split(self.key_random)
            self.latitude_sun += (
                jax.random.normal(
                    key=subkey,
                    shape=(),
                )
                * self.height
                / 2
                / jax.numpy.sqrt(self.period_sun)
            )
        elif self.method_sun == "sine":
            self.latitude_sun = self.height // 2 + self.height // 2 * jax.numpy.sin(
                2 * jax.numpy.pi * self.timestep / self.period_sun
            )
        elif self.method_sun == "linear":
            self.latitude_sun = (
                self.height // 2 + self.height * self.timestep // self.period_sun
            )
        else:
            raise ValueError(f"Unknown method_sun: {self.method_sun}")
        self.latitude_sun = jax.numpy.round(self.latitude_sun).astype(jnp.int32)
        self.latitude_sun = self.latitude_sun % self.height
        self.map = self.map.at[self.latitude_sun, :, 0].set(1.0)
