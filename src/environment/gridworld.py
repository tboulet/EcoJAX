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
    agents_positions: jnp.ndarray  # (n_max_agents, 2)
    agents_exist: jnp.ndarray  # (n_max_agents,)


class GridworldEnv:
    """The Gridworld environment."""

    name_channel_to_idx = {
        "sun": 0,
        "plants": 1,
        "agents": 2,
    }
    n_channels_map = len(name_channel_to_idx)

    def __init__(
        self,
        config: Dict[str, Any],
        n_agents_max: int,
        n_agents_initial: int,
    ) -> None:
        """Initialize an instance of the Gridworld environment. This class allows to deal in a comprehensive way with a Gridworld environment
        that represents the world with which the agents interact. It is purely agnostic of the agents and their learning algorithms.

        In order to apply JAX transformation to such an OOP class, the following principles are applied:
        - the environmental parameters than are not changing during the simulation are stored in the class attributes, e.g. self.width, self.period_sun, etc.
        - the environmental objects that will change of value through the simulation are stored in the state, which is an object from a class inheriting the flax.struct.dataclass,
          which allows to apply JAX transformations to the object.

        The pipeline of the environment is the following:
        >>> env = GridworldEnv(config, n_agents_max, n_agents_initial)
        >>> state_env, observations_agents = env.start(key_random)
        >>> while not done:

        Args:
            config (Dict[str, Any]): the configuration of the environment
            n_agents_max (int): the maximum number of agents the environment can handle
            n_agents_initial (int): the initial number of agents in the environment. They are for now randomly placed in the environment.
        """
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
        # plants Dynamics
        self.proportion_plant_initial = config["proportion_plant_initial"]
        self.logit_p_base_plant_growth = logit(config["p_base_plant_growth"])
        self.logit_p_base_plant_death = logit(config["p_base_plant_death"])
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
        # Agents Parameters
        self.n_agents_max = n_agents_max
        self.n_agents_initial = n_agents_initial
        assert (
            self.n_agents_initial <= self.n_agents_max
        ), "n_agents_initial must be less than or equal to n_agents_max"
        self.vision_range_agent = config["vision_range_agent"]

    def start(
        self,
        key_random: jnp.ndarray,
    ) -> EnvState:
        """Start the environment. This initialize the state and also returns the initial observations of the agents.

        Args:
            key_random (jnp.ndarray): the random key used for the initialization

        Returns:
            EnvState: the initial state of the environment
            jnp.ndarray: the observations of the agents after the initialization, of shape (n_max_agents, dim_observation)
        """
        idx_sun = self.name_channel_to_idx["sun"]
        idx_plants = self.name_channel_to_idx["plants"]
        idx_agents = self.name_channel_to_idx["agents"]
        # Initialize the map
        map = jnp.zeros(shape=(self.height, self.width, self.n_channels_map))
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
            map = map.at[:, :, idx_sun].set(effect_map)
        else:
            latitude_sun = None
        # Initialize the plants
        key_random, subkey = jax.random.split(key_random)
        map = map.at[:, :, idx_plants].set(
            jax.random.bernoulli(
                key=subkey,
                p=self.proportion_plant_initial,
                shape=(self.height, self.width),
            )
        )
        # Initialize the agents
        key_random, subkey1, subkey2 = jax.random.split(key_random, 3)
        agents_exist = jnp.zeros(self.n_agents_max)
        agents_exist = agents_exist.at[: self.n_agents_initial].set(1)
        agents_positions = jnp.zeros((self.n_agents_max, 2), dtype=jnp.int32)
        agents_positions = agents_positions.at[: self.n_agents_initial, 0].set(
            jax.random.randint(
                key=subkey1,
                minval=0,
                maxval=self.height,
                shape=(self.n_agents_initial,),
            ),
        )
        agents_positions = agents_positions.at[: self.n_agents_initial, 1].set(
            jax.random.randint(
                key=subkey2,
                minval=0,
                maxval=self.width,
                shape=(self.n_agents_initial,),
            ),
        )
        map = map.at[agents_positions[:, 0], agents_positions[:, 1], idx_agents].set(1)
        # Return the initial state and observations
        state = EnvState(
            timestep=0,
            map=map,
            latitude_sun=latitude_sun,
            agents_positions=agents_positions,
            agents_exist=agents_exist,
        )
        observations_agents = self.get_observations_agents(state=state)
        return state, observations_agents

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key_random: jnp.ndarray,
        state: EnvState,
        actions: jnp.ndarray,
    ) -> Tuple[
        EnvState,
        jnp.ndarray,
        bool,
        Dict[str, Any],
    ]:
        """Performe one step of the Gridworld environment.

        Args:
            key_random (jnp.ndarray): the random key used for this step
            jnp.ndarray: the observations to give to the agents, of shape (n_max_agents, dim_observation)
            state (EnvState): the state of the environment
            actions (jnp.ndarray): the actions to perform

        Returns:
            EnvState: the new state of the environment
            jnp.ndarray: the observations of the agents after the step, of shape (n_max_agents, dim_observation)
            bool: whether the environment is done
            Dict[str, Any]: the info of the environment
        """
        idx_sun = self.name_channel_to_idx["sun"]
        idx_plants = self.name_channel_to_idx["plants"]
        # Update the sun
        key_random, subkey = jax.random.split(key_random)
        state = self.update_sun(state=state, key_random=subkey)
        # Grow plants
        map_plants = state.map[:, :, self.name_channel_to_idx["plants"]]
        map_n_plant_in_radius_plant_reproduction = convolve2d(
            map_plants,
            self.kernel_plant_reproduction,
            mode="same",
        )
        map_n_plant_in_radius_plant_asphyxia = convolve2d(
            map_plants,
            self.kernel_plant_asphyxia,
            mode="same",
        )
        map_sun = state.map[:, :, idx_sun]
        map_plants_probs = sigmoid(
            x=self.logit_p_base_plant_growth * (1 - map_plants)
            + (1 - self.logit_p_base_plant_death) * map_plants
            + self.factor_sun_effect * map_sun
            + self.factor_plant_reproduction * map_n_plant_in_radius_plant_reproduction
            - self.factor_plant_asphyxia * map_n_plant_in_radius_plant_asphyxia
        )
        key_random, subkey = jax.random.split(key_random)
        map_plants = jax.random.bernoulli(
            key=subkey,
            p=map_plants_probs,
            shape=map_plants.shape,
        )
        # Update the state
        state = state.replace(
            timestep=state.timestep + 1,
            map=state.map.at[:, :, idx_plants].set(map_plants),
        )
        observations_agents = self.get_observations_agents(state=state)
        return (
            state,
            observations_agents,
            False,
            {},
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_RGB_map(self, state: EnvState) -> Any:
        """A function for rendering the environment. It returns the RGB map of the environment.

        Args:
            state (EnvState): the state of the environment

        Returns:
            Any: the RGB map of the environment
        """
        return state.map[:, :, :3]

    # ================== Helper functions ==================

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
        # Update the latitude of the sun depending on the method
        idx_sun = self.name_channel_to_idx["sun"]
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
            map=state.map.at[:, :, idx_sun].set(
                jnp.roll(state.map[:, :, idx_sun], shift, axis=0)
            ),
        )

    def get_observations_agents(self, state: EnvState) -> jnp.ndarray:
        """Extract the observations of the agents from the state of the environment.

        Args:
            state (EnvState): the state of the environment

        Returns:
            jnp.ndarray: the observations of the agents, of shape (n_max_agents, dim_observation)
        """
        
        # Compute the observations of shape (n_max_agents, 5, 5)
        padded_map = jnp.pad(state.map, pad_width=self.vision_range_agent, mode='constant', constant_values=0)
        padded_coords = state.agents_positions + self.vision_range_agent
        
        # Create a grid of indices for the vision range
        grid_x, grid_y = jnp.meshgrid(jnp.arange(-self.vision_range_agent, self.vision_range_agent + 1), 
                                    jnp.arange(-self.vision_range_agent, self.vision_range_agent + 1), indexing='ij')

        # Add the grid indices to the agent coordinates to get the observation indices
        agent_x = padded_coords[:, 0][:, jnp.newaxis, jnp.newaxis] + grid_x
        agent_y = padded_coords[:, 1][:, jnp.newaxis, jnp.newaxis] + grid_y

        # Extract the observations
        observations = padded_map[agent_x, agent_y]

        return observations