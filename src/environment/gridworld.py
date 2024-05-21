# Gridworld EcoJAX environment

from functools import partial
from time import sleep
from typing import Any, Dict, Tuple
import jax.numpy as jnp
import jax
from jax.scipy.signal import convolve2d
from flax import struct

from src.environment.base_env import BaseEnvironment
from src.utils import sigmoid, logit


@struct.dataclass
class EnvState:
    timestep: int
    map: jnp.ndarray  # (height, width, dim_tile)
    latitude_sun: int


class GridworldEnv:
    """The Gridworld environment."""

    def __init__(self, config) -> None:
        self.config = config
        # Environment Parameters
        self.width = config["width"]
        self.height = config["height"]
        self.type_border = config["type_border"]
        # Sun Parameters
        self.period_sun = config["period_sun"]
        self.method_sun = config["method_sun"]
        self.radius_sun_effect = config["radius_sun_effect"]
        self.radius_sun_perception = config["radius_sun_perception"]
        # Food Dynamics
        self.proportion_plant_initial = config["proportion_plant_initial"]
        self.logit_p_base_plant_growth = logit(config["p_base_plant_growth"])
        self.logit_p_base_plant_death = logit(config["p_base_die_food"])
        self.factor_sun_effect = config["factor_sun_effect"]
        self.factor_plant_reproduction = config["factor_plant_reproduction"]
        self.radius_plant_reproduction = config["radius_plant_reproduction"]
        self.kernel_plant_reproduction = jnp.ones(
            (
                config["radius_plant_reproduction"],
                config["radius_plant_reproduction"],
            )
        ) / (config["radius_plant_reproduction"] ** 2)
        self.factor_plant_asphyxia = config["factor_plant_asphyxia"]
        self.radius_plant_asphyxia = config["radius_plant_asphyxia"]
        self.kernel_plant_asphyxia = jnp.ones(
            (config["radius_plant_asphyxia"], config["radius_plant_asphyxia"])
        ) / (config["radius_plant_asphyxia"] ** 2)

    def start(
        self,
        key_random: jnp.ndarray,
    ) -> EnvState:
        # Initialize the map
        map = jnp.zeros(shape=(self.height, self.width, 3))
        # Initialize the food
        map = map.at[:, :, 1].set(
            jax.random.bernoulli(
                key=key_random,
                p=self.proportion_plant_initial,
                shape=(self.height, self.width),
            )
        )
        # Initialize the sun
        if self.method_sun != "none":
            latitude_sun = self.height // 2
            latitudes = jnp.arange(self.height)
            distance_from_sun = jnp.minimum(
                jnp.abs(latitudes - latitude_sun),
                self.height - jnp.abs(latitudes - latitude_sun),
            )
            effect = jnp.clip(1 - distance_from_sun / self.radius_sun_effect, 0, 1)
            effect_map = jnp.repeat(effect[:, None], self.width, axis=1)
            map = map.at[:, :, 0].set(effect_map)
        else:
            latitude_sun = None
        # Return the initial state
        state = EnvState(
            timestep=0,
            map=map,
            latitude_sun=latitude_sun,
        )
        return state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key_random: jnp.ndarray,
        state: EnvState,
        actions: jnp.ndarray,
    ) -> Tuple[
        EnvState,
        bool,
        Dict[str, Any],
    ]:
        """Performe one step of the Gridworld environment.

        Args:
            key_random (jnp.ndarray): the random key used for this step
            state (EnvState): the state of the environment
            actions (jnp.ndarray): the actions to perform

        Returns:
            EnvState: the new state of the environment
            bool: whether the environment is done
            Dict[str, Any]: the info of the environment
        """
        # Update the sun
        key_random, subkey = jax.random.split(key_random)
        state = self.update_sun(state=state, key_random=subkey)
        # Grow food
        map_food = state.map[:, :, 1]  # (W, H)
        map_n_plant_in_radius_plant_reproduction = convolve2d(
            map_food,
            self.kernel_plant_reproduction,
            mode="same",
        )
        map_n_plant_in_radius_plant_asphyxia = convolve2d(
            map_food,
            self.kernel_plant_asphyxia,
            mode="same",
        )
        map_food_probs = sigmoid(
            x=self.logit_p_base_plant_growth * (1 - map_food)
            # + self.factor_sun_grow_food * map_is_sunny
            + self.factor_plant_reproduction * map_n_plant_in_radius_plant_reproduction
            - self.factor_plant_asphyxia * map_n_plant_in_radius_plant_asphyxia
            - self.logit_p_base_plant_death * map_food
        )
        key_random, subkey = jax.random.split(key_random)
        map_food = jax.random.bernoulli(
            key=subkey,
            p=map_food_probs,
            shape=map_food.shape,
        )
        # Update the state and return it
        state = state.replace(
            timestep=state.timestep + 1,
            map=state.map.at[:, :, 1].set(map_food),
        )
        return state, False, {}

    def get_RGB_map(self, state: EnvState) -> Any:
        return state.map

    # Helper functions

    def update_sun(self, state: EnvState, key_random: jnp.ndarray) -> EnvState:
        """Change the state concerning the sun dynamics.
        The sun new value depends on the method_sun argument.

        Args:
            state (EnvState): the current state of the environment

        Raises:
            ValueError: if the method_sun is unknown

        Returns:
            EnvState: the new state of the environment
        """
        # Remove the red coloration of the sun
        # self.map = self.map.at[self.latitude_sun, :, 0].set(0.0)
        # Update the latitude of the sun depending on the method
        if self.method_sun == "none":
            return state
        elif self.method_sun == "fixed":
            return state
        elif self.method_sun == "random":
            latitude_sun = jax.random.randint(
                key=key_random,
                minval=0,
                maxval=self.height,
                shape=(),
            )
        elif self.method_sun == "brownian":
            latitude_sun = state.latitude_sun + (
                jax.random.normal(
                    key=key_random,
                    shape=(),
                )
                * self.height
                / 2
                / jax.numpy.sqrt(self.period_sun)
            )
        elif self.method_sun == "sine":
            latitude_sun = self.height // 2 + self.height // 2 * jax.numpy.sin(
                2 * jax.numpy.pi * state.timestep / self.period_sun
            )
        elif self.method_sun == "linear":
            latitude_sun = (
                self.height // 2 + self.height * state.timestep // self.period_sun
            )
        else:
            raise ValueError(f"Unknown method_sun: {self.method_sun}")
        latitude_sun = jax.numpy.round(latitude_sun).astype(jnp.int32)
        latitude_sun = latitude_sun % self.height
        shift = latitude_sun - state.latitude_sun
        return state.replace(
            latitude_sun=latitude_sun,
            map=state.map.at[:, :, 0].set(jnp.roll(state.map[:, :, 0], shift, axis=0)),
        )
