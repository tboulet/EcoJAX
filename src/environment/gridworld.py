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
    """This JAX data class represents the state of the environment. It is used to store the state of the environment and to apply JAX transformations to it.
    Instances of this class represents objects that will change of value through the simulation and that entirely representing the non-constant part of the environment.
    """

    # The current timestep of the environment
    timestep: int

    # The current map of the environment, of shape (H, W, C) where C is the number of channels used to represent the environment
    map: jnp.ndarray  # (height, width, dim_tile) in R

    # The latitude of the sun (the row of the map where the sun is). It represents entirely the sun location.
    latitude_sun: int

    # Where the agents are, of shape (n_max_agents, 2). positions_agents[i, :] represents the (x,y) coordinates of the i-th agent in the map. Ghost agents are still represented in the array (in position (0,0)).
    positions_agents: jnp.ndarray  # (n_max_agents, 2) in [0, height-1] x [0, width-1]
    # The orientation of the agents, of shape (n_max_agents,) and of values in {0, 1, 2, 3}. orientation_agents[i] represents the index of its orientation in the env.
    # The orientation of an agent will have an impact on the way the agent's surroundings are perceived, because a certain rotation will be performed on the agent's vision in comparison to the traditional map[x-v:x+v+1, y-v:y+v+1, :] vision.
    # The angle the agent is facing is given by orientation_agents[i] * 90 degrees (modulo 360 degrees), where 0 is facing north.
    orientation_agents: jnp.ndarray  # (n_max_agents,) in {0, 1, 2, 3}
    # Whether the agents exist or not, of shape (n_max_agents,) and of values in {0, 1}. are_existing_agents[i] represents whether the i-th agent actually exists in the environment and should interact with it.
    # An non existing agent is called a Ghost Agent and is only kept as a placeholder in the positions_agents array, in order to keep the array of positions_agents of shape (n_max_agents, 2).
    are_existing_agents: jnp.ndarray  # (n_max_agents,) in {0, 1}


@struct.dataclass
class AgentObservation:
    """This class represents the observation of an agent. It is used to store the observation of an agent and to apply JAX transformations to it."""

    # The visual field of the agent, of shape (2v+1, 2v+1, n_channels_map) where n_channels_map is the number of channels used to represent the environment.
    visual_field: jnp.ndarray  # (2v+1, 2v+1, n_channels_map) in R


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
        >>>     actions = ...
        >>>     state_env, observations_agents, done, info = env.step(key_random, state_env, actions)
        >>>     RGB_map = env.get_RGB_map(state_env)

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
        # Create a grid of indices for the vision range
        self.grid_indexes_vision_x, self.grid_indexes_vision_y = jnp.meshgrid(
            jnp.arange(-self.vision_range_agent, self.vision_range_agent + 1),
            jnp.arange(-self.vision_range_agent, self.vision_range_agent + 1),
            indexing="ij",
        )

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
        H, W = self.height, self.width
        # Initialize the map
        map = jnp.zeros(shape=(H, W, self.n_channels_map))
        # Initialize the sun
        if self.method_sun != "none":
            latitude_sun = H // 2
            latitudes = jnp.arange(H)
            distance_from_sun = jnp.minimum(
                jnp.abs(latitudes - latitude_sun),
                H - jnp.abs(latitudes - latitude_sun),
            )
            effect = jnp.clip(1 - distance_from_sun / self.radius_sun_effect, 0, 1)
            effect_map = jnp.repeat(effect[:, None], W, axis=1)
            map = map.at[:, :, idx_sun].set(effect_map)
        else:
            latitude_sun = None
        # Initialize the plants
        key_random, subkey = jax.random.split(key_random)
        map = map.at[:, :, idx_plants].set(
            jax.random.bernoulli(
                key=subkey,
                p=self.proportion_plant_initial,
                shape=(H, W),
            )
        )
        # Initialize the agents
        key_random, subkey = jax.random.split(key_random)
        positions_agents = jax.random.randint(
            key=subkey,
            shape=(self.n_agents_max, 2),
            minval=0,
            maxval=max(H, W),
        )
        positions_agents %= jnp.array([H, W])
        map = map.at[positions_agents[:, 0], positions_agents[:, 1], idx_agents].set(1)
        key_random, subkey = jax.random.split(key_random)
        orientation_agents = jax.random.randint(
            key=subkey,
            shape=(self.n_agents_max,),
            minval=0,
            maxval=4,
        )
        are_existing_agents = jnp.zeros(self.n_agents_max)
        are_existing_agents = are_existing_agents.at[: self.n_agents_initial].set(1)
        # Return the initial state and observations
        state = EnvState(
            timestep=0,
            map=map,
            latitude_sun=latitude_sun,
            positions_agents=positions_agents,
            orientation_agents=orientation_agents,
            are_existing_agents=are_existing_agents,
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
        # Update the timestep
        state = state.replace(timestep=state.timestep + 1)
        # Apply the actions of the agents
        key_random, subkey = jax.random.split(key_random)
        state = self.step_action_agents(state=state, actions=actions, key_random=subkey)
        # Update the sun
        key_random, subkey = jax.random.split(key_random)
        state = self.step_update_sun(state=state, key_random=subkey)
        # Grow plants
        key_random, subkey = jax.random.split(key_random)
        state = self.step_grow_plants(state=state, key_random=subkey)
        # Update the state
        observations_agents = self.get_observations_agents(state=state)
        # Return the new state and observations
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

    def step_grow_plants(self, state: EnvState, key_random: jnp.ndarray) -> jnp.ndarray:
        """Modify the state of the environment by growing the plants."""
        idx_sun = self.name_channel_to_idx["sun"]
        idx_plants = self.name_channel_to_idx["plants"]
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
        return state.replace(map=state.map.at[:, :, idx_plants].set(map_plants))

    def step_update_sun(self, state: EnvState, key_random: jnp.ndarray) -> EnvState:
        """Modify the state of the environment by updating the sun.
        The method of updating the sun is defined by the attribute self.method_sun.
        """
        # Update the latitude of the sun depending on the method
        idx_sun = self.name_channel_to_idx["sun"]
        H, W, C = state.map.shape
        if self.method_sun == "none":
            return state
        elif self.method_sun == "fixed":
            return state
        elif self.method_sun == "random":
            latitude_sun = jax.random.randint(
                key=key_random,
                minval=0,
                maxval=H,
                shape=(),
            )
        elif self.method_sun == "brownian":
            latitude_sun = state.latitude_sun + (
                jax.random.normal(
                    key=key_random,
                    shape=(),
                )
                * H
                / 2
                / jax.numpy.sqrt(self.period_sun)
            )
        elif self.method_sun == "sine":
            latitude_sun = H // 2 + H // 2 * jax.numpy.sin(
                2 * jax.numpy.pi * state.timestep / self.period_sun
            )
        elif self.method_sun == "linear":
            latitude_sun = (
                H // 2 + H * state.timestep // self.period_sun
            )
        else:
            raise ValueError(f"Unknown method_sun: {self.method_sun}")
        latitude_sun = jax.numpy.round(latitude_sun).astype(jnp.int32)
        latitude_sun = latitude_sun % H
        shift = latitude_sun - state.latitude_sun
        return state.replace(
            latitude_sun=latitude_sun,
            map=state.map.at[:, :, idx_sun].set(
                jnp.roll(state.map[:, :, idx_sun], shift, axis=0)
            ),
        )

    def step_action_agents(
        self, state: EnvState, actions: jnp.ndarray, key_random: jnp.ndarray
    ) -> EnvState:
        """Modify the state of the environment by applying the actions of the agents."""
        H, W, C = state.map.shape
        idx_agents = self.name_channel_to_idx["agents"]
        # Apply the movements of the agents

        def get_single_agent_new_position_and_orientation(
            agent_position: jnp.ndarray,
            agent_orientation: jnp.ndarray,
            action: jnp.ndarray,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Get the new position and orientation of a single agent.
            Args:
                agent_position (jnp.ndarray): the position of the agent, of shape (2,)
                agent_orientation (jnp.ndarray): the orientation of the agent, of shape ()
                action (jnp.ndarray): the action of the agent, of shape ()

            Returns:
                jnp.ndarray: the new position of the agent, of shape (2,)
                jnp.ndarray: the new orientation of the agent, of shape ()
            """
            # Compute the new position and orientation of the agent
            agent_orientation_new = (agent_orientation + action) % 4
            angle_new = agent_orientation_new * jnp.pi / 2
            d_position = jnp.array([jnp.cos(angle_new), jnp.sin(angle_new)]).astype(
                jnp.int32
            )
            agent_position_new = agent_position + d_position
            agent_position_new = agent_position_new % jnp.array([H, W])

            # Warning : if non-moving action are implemented, this would require a jnp.select here

            # Return the new position and orientation of the agent
            return agent_position_new, agent_orientation_new

        # Vectorize the function to get the new position and orientation of many agents
        get_many_agents_new_position_and_orientation = jax.vmap(
            get_single_agent_new_position_and_orientation, in_axes=(0, 0, 0)
        )

        # Compute the new positions and orientations of all the agents
        positions_agents_new, orientation_agents_new = (
            get_many_agents_new_position_and_orientation(
                state.positions_agents,
                state.orientation_agents,
                actions,
            )
        )
        
        # Update the map with the new positions of the agents (remove the agents from their previous positions and add them to their new positions)
        map_new = state.map.at[
            state.positions_agents[:, 0], state.positions_agents[:, 1], idx_agents
        ].set(0)
        map_new = map_new.at[
            positions_agents_new[:, 0], positions_agents_new[:, 1], idx_agents
        ].set(1)

        # Update the state
        return state.replace(
            map=map_new,
            positions_agents=positions_agents_new,
            orientation_agents=orientation_agents_new,
        )

    def get_observations_agents(self, state: EnvState) -> jnp.ndarray:
        """Extract the observations of the agents from the state of the environment.

        Args:
            state (EnvState): the state of the environment

        Returns:
            jnp.ndarray: the observations of the agents, of shape (n_max_agents, dim_observation)
            with dim_observation = (2v+1, 2v+1, n_channels_map)
        """

        def get_single_agent_obs(
            agent_position: jnp.ndarray,
            agent_orientation: jnp.ndarray,
        ) -> jnp.ndarray:
            """Get the observation of a single agent.

            Args:
                state (EnvState): the state of the environment
                agent_position (jnp.ndarray): the position of the agent, of shape (2,)
                agent_orientation (jnp.ndarray): the orientation of the agent, of shape ()

            Returns:
                jnp.ndarray: the observation of the agent, of shape (2v+1, 2v+1, n_channels_map)
            """
            H, W, c = state.map.shape

            # Get the visual field of the agent
            visual_field_x = agent_position[0] + self.grid_indexes_vision_x
            visual_field_y = agent_position[1] + self.grid_indexes_vision_y
            obs = state.map[
                visual_field_x % H,
                visual_field_y % W,
            ]
            # Rotate the visual field according to the orientation of the agent
            obs = jnp.select(
                [
                    agent_orientation == 0,
                    agent_orientation == 1,
                    agent_orientation == 2,
                    agent_orientation == 3,
                ],
                [
                    obs,
                    jnp.rot90(obs, k=1, axes=(0, 1)),
                    jnp.rot90(obs, k=2, axes=(0, 1)),
                    jnp.rot90(obs, k=3, axes=(0, 1)),
                ],
            )
            # Return the observation
            return obs

        # Vectorize the function to get the observation of many agents
        get_many_agents_obs = jax.vmap(get_single_agent_obs, in_axes=(0, 0))

        # Compute the observations of all the agents
        observations = get_many_agents_obs(
            state.positions_agents,
            state.orientation_agents,
        )

        # print(f"Map : {state.map[..., 0]}")
        # print(f"Agents positions : {state.positions_agents}")
        # print(f"Agents orientations : {state.orientation_agents}")
        # print(f"First agent obs : {observations[0, ..., 0]}, shape : {observations[0, ..., 0].shape}")
        # raise ValueError("Stop")

        return observations
