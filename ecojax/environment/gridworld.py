# Gridworld EcoJAX environment

from collections import defaultdict
from functools import partial
import os
from time import sleep
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.scipy.signal import convolve2d
from flax.struct import PyTreeNode, dataclass
from jax.debug import breakpoint as jbreakpoint
from tqdm import tqdm

from ecojax.core.eco_info import EcoInformation
from ecojax.environment import EcoEnvironment
from ecojax.metrics.aggregators import Aggregator
from ecojax.spaces import EcojaxSpace, DictSpace, DiscreteSpace, ContinuousSpace
from ecojax.types import ActionAgent, ObservationAgent, StateEnv, StateSpecies
from ecojax.utils import (
    DICT_COLOR_TAG_TO_RGB,
    instantiate_class,
    jprint,
    jprint_and_breakpoint,
    sigmoid,
    logit,
    try_get,
)
from ecojax.video import VideoRecorder


@dataclass
class AgentGriworld:
    # Where the agents are, of shape (n_max_agents, 2). positions_agents[i, :] represents the (x,y) coordinates of the i-th agent in the map. Ghost agents are still represented in the array (in position (0,0)).
    positions_agents: jnp.ndarray  # (n_max_agents, 2) in [0, height-1] x [0, width-1]
    # The orientation of the agents, of shape (n_max_agents,) and of values in {0, 1, 2, 3}. orientation_agents[i] represents the index of its orientation in the env.
    # The orientation of an agent will have an impact on the way the agent's surroundings are perceived, because a certain rotation will be performed on the agent's vision in comparison to the traditional map[x-v:x+v+1, y-v:y+v+1, :] vision.
    # The angle the agent is facing is given by orientation_agents[i] * 90 degrees (modulo 360 degrees), where 0 is facing north.
    orientation_agents: jnp.ndarray  # (n_max_agents,) in {0, 1, 2, 3}
    # Whether the agents exist or not, of shape (n_max_agents,) and of values in {0, 1}. are_existing_agents[i] represents whether the i-th agent actually exists in the environment and should interact with it.
    # An non existing agent is called a Ghost Agent and is only kept as a placeholder in the positions_agents array, in order to keep the array of positions_agents of shape (n_max_agents, 2).
    are_existing_agents: jnp.ndarray  # (n_max_agents,) in {0, 1}
    # The energy level of the agents, of shape (n_max_agents,). energy_agents[i] represents the energy level of the i-th agent.
    energy_agents: jnp.ndarray  # (n_max_agents,) in [0, +inf)
    # The age of the agents
    age_agents: jnp.ndarray  # (n_max_agents,) in [0, +inf)
    # The appearance of the agent, encoded as a vector in R^dim_appearance. appearance_agents[i, :] represents the appearance of the i-th agent.
    # The appearance of an agent allows the agents to distinguish their genetic proximity, as agents with similar appearances are more likely to be genetically close.
    # By convention : a non-agent has an appearance of zeros, the common ancestors have an appearance of ones, and m superposed agents have an appearance of their average.
    appearance_agents: jnp.ndarray  # (n_max_agents, dim_appearance) in R


@dataclass
class VideoMemory:
    # The current frame of the video
    idx_end_of_video: int

    # The last


@dataclass
class StateEnvGridworld(StateEnv):
    # The current timestep of the environment
    timestep: int

    # The current map of the environment, of shape (H, W, C) where C is the number of channels used to represent the environment
    map: jnp.ndarray  # (height, width, dim_tile) in R

    # The latitude of the sun (the row of the map where the sun is). It represents entirely the sun location.
    latitude_sun: int

    # The state of the agents in the environment
    agents: AgentGriworld  # Batched

    # The lifespan and population aggregators
    metrics_lifespan: List[PyTreeNode]
    metrics_population: List[PyTreeNode]

    # The last n_steps_per_video frames of the video
    video: jnp.ndarray  # (n_steps_per_video, height, width, 3) in [0, 1]


class GridworldEnv(EcoEnvironment):
    """A Gridworld environment."""

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
        >>> (
        >>>     observations_agents,
        >>>     eco_information,
        >>>     done,
        >>>     info,
        >>> ) = env.reset(key_random)
        >>>
        >>> while not done:
        >>>
        >>>     env.render()
        >>>
        >>>     actions = ...
        >>>
        >>>     key_random, subkey = random.split(key_random)
        >>>     (
        >>>         observations_agents,
        >>>         eco_information,
        >>>         done_env,
        >>>         info_env,
        >>>     ) = env.step(
        >>>         key_random=subkey,
        >>>         actions=actions,
        >>>     )
        >>>

        Args:
            config (Dict[str, Any]): the configuration of the environment
            n_agents_max (int): the maximum number of agents the environment can handle
            n_agents_initial (int): the initial number of agents in the environment. They are for now randomly placed in the environment.
        """
        self.config = config
        self.n_agents_max = n_agents_max
        self.n_agents_initial = n_agents_initial
        assert (
            self.n_agents_initial <= self.n_agents_max
        ), "n_agents_initial must be less than or equal to n_agents_max"

        # Environment Parameters
        self.width: int = config["width"]
        self.height: int = config["height"]
        self.is_terminal: bool = config["is_terminal"]
        self.allow_multiple_agents_per_tile = config.get(
            "allow_multiple_agents_per_tile", True
        )
        self.period_logging: int = int(max(1, self.config["period_logging"]))
        self.list_names_channels: List[str] = ["sun", "plants", "agents", "agent_ages"]
        self.list_names_channels += [
            f"appearance_{i}" for i in range(config["dim_appearance"])
        ]
        self.dict_name_channel_to_idx: Dict[str, int] = {
            name_channel: idx_channel
            for idx_channel, name_channel in enumerate(self.list_names_channels)
        }
        self.n_channels_map: int = len(self.dict_name_channel_to_idx)
        self.list_indexes_channels_visual_field: List[int] = []
        for name_channel in config["list_channels_visual_field"]:
            assert name_channel in self.dict_name_channel_to_idx, "Channel not found"
            self.list_indexes_channels_visual_field.append(
                self.dict_name_channel_to_idx[name_channel]
            )
        self.dict_name_channel_to_idx_visual_field: Dict[str, int] = {
            name_channel: idx_channel
            for idx_channel, name_channel in enumerate(config["list_channels_visual_field"])
        }
        self.n_channels_visual_field: int = len(self.list_indexes_channels_visual_field)

        # Metrics parameters
        self.names_measures: List[str] = sum(
            [names for type_measure, names in config["metrics"]["measures"].items()], []
        )

        # Video parameters
        self.cfg_video = config["metrics"]["config_video"]
        self.do_video: bool = self.cfg_video["do_video"]
        self.n_steps_per_video: int = self.cfg_video["n_steps_per_video"]
        self.fps_video: int = self.cfg_video["fps_video"]
        self.dir_videos: str = self.cfg_video["dir_videos"]
        self.height_max_video: int = self.cfg_video["height_max_video"]
        self.width_max_video: int = self.cfg_video["width_max_video"]
        self.dict_name_channel_to_color_tag: Dict[str, str] = self.cfg_video[
            "dict_name_channel_to_color_tag"
        ]
        self.color_tag_background = try_get(
            self.config, "color_background", default="white"
        )
        self.color_tag_unknown_channel = try_get(
            self.config, "color_unknown_channel", default="black"
        )
        self.dict_idx_channel_to_color_tag: Dict[int, str] = {}

        os.makedirs(self.dir_videos, exist_ok=True)

        for name_channel, idx_channel in self.dict_name_channel_to_idx.items():
            if name_channel in self.dict_name_channel_to_color_tag:
                self.dict_idx_channel_to_color_tag[idx_channel] = (
                    self.dict_name_channel_to_color_tag[name_channel]
                )
            else:
                self.dict_idx_channel_to_color_tag[idx_channel] = (
                    self.color_tag_unknown_channel
                )

        # Sun Parameters
        self.period_sun: int = config["period_sun"]
        self.method_sun: str = config["method_sun"]
        self.radius_sun_effect: int = config["radius_sun_effect"]
        self.radius_sun_perception: int = config["radius_sun_perception"]

        # Plants Dynamics
        self.proportion_plant_initial: float = config["proportion_plant_initial"]
        self.logit_p_base_plant_growth: float = logit(config["p_base_plant_growth"])
        self.logit_p_base_plant_death: float = logit(config["p_base_plant_death"])
        self.factor_sun_effect: float = config["factor_sun_effect"]
        self.factor_plant_reproduction: float = config["factor_plant_reproduction"]
        self.radius_plant_reproduction: int = config["radius_plant_reproduction"]
        self.kernel_plant_reproduction = jnp.ones(
            (
                config["radius_plant_reproduction"],
                config["radius_plant_reproduction"],
            )
        ) / (config["radius_plant_reproduction"] ** 2)
        self.factor_plant_asphyxia: float = config["factor_plant_asphyxia"]
        self.radius_plant_asphyxia: int = config["radius_plant_asphyxia"]
        self.kernel_plant_asphyxia = jnp.ones(
            (config["radius_plant_asphyxia"], config["radius_plant_asphyxia"])
        ) / (config["radius_plant_asphyxia"] ** 2)

        # ======================== Agent Parameters ========================

        # Observations
        self.list_observations: List[str] = config["list_observations"]
        assert (
            len(self.list_observations) > 0
        ), "The list of observations must be non-empty"

        @dataclass
        class ObservationAgentGridworld(ObservationAgent):
            # The visual field of the agent, of shape (2v+1, 2v+1, n_channels_map) where n_channels_map is the number of channels used to represent the environment.
            if "visual_field" in self.list_observations:
                visual_field: jnp.ndarray  # (2v+1, 2v+1, n_channels_map) in R

            # The energy level of an agent, of shape () and in [0, +inf).
            if "energy" in self.list_observations:
                energy: jnp.ndarray

            # The age of an agent, of shape () and in [0, +inf).
            if "age" in self.list_observations:
                age: jnp.ndarray

        self.ObservationAgentGridworld = ObservationAgentGridworld

        # Create the observation space
        observation_dict = {}
        if "visual_field" in self.list_observations:
            self.vision_range_agent: int = config["vision_range_agent"]
            self.grid_indexes_vision_x, self.grid_indexes_vision_y = jnp.meshgrid(
                jnp.arange(-self.vision_range_agent, self.vision_range_agent + 1),
                jnp.arange(-self.vision_range_agent, self.vision_range_agent + 1),
                indexing="ij",
            )
            observation_dict["visual_field"] = ContinuousSpace(
                shape=(
                    2 * self.vision_range_agent + 1,
                    2 * self.vision_range_agent + 1,
                    self.n_channels_visual_field,
                ),
                low=None,
                high=None,
            )
        if "energy" in self.list_observations:
            observation_dict["energy"] = ContinuousSpace(shape=(), low=0, high=None)
        if "age" in self.list_observations:
            observation_dict["age"] = ContinuousSpace(shape=(), low=0, high=None)
        self.observation_space = DictSpace(observation_dict)

        # Actions
        self.list_actions: List[str] = config["list_actions"]
        assert len(self.list_actions) > 0, "The list of actions must be non-empty"
        self.action_to_idx: Dict[str, int] = {
            action: idx for idx, action in enumerate(self.list_actions)
        }
        self.n_actions = len(self.list_actions)

        # Agent's internal dynamics
        self.age_max: int = config["age_max"]
        self.energy_max: float = config["energy_max"]
        self.energy_initial: float = config["energy_initial"]
        self.energy_loss_idle: float = config["energy_loss_idle"]
        self.energy_loss_action: float = config["energy_loss_action"]
        self.energy_food: float = config["energy_food"]
        self.energy_thr_death: float = config["energy_thr_death"]
        self.energy_req_reprod: float = config["energy_req_reprod"]
        self.energy_cost_reprod: float = config["energy_cost_reprod"]
        self.energy_transfer_loss: float = config.get("energy_transfer_loss", 0.0)
        self.energy_transfer_gain: float = config.get("energy_transfer_gain", 0.0)
        self.move_prob_initial: float = config.get("move_prob_initial", 1.0)
        self.move_prob_gradient: float = config.get("move_prob_gradient", 0.0)
        # Other
        self.fill_value: int = self.n_agents_max

    def reset(
        self,
        key_random: jnp.ndarray,
    ) -> Tuple[
        StateEnvGridworld,
        ObservationAgent,
        EcoInformation,
        bool,
        Dict[str, Any],
    ]:
        idx_sun = self.dict_name_channel_to_idx["sun"]
        idx_plants = self.dict_name_channel_to_idx["plants"]
        idx_agents = self.dict_name_channel_to_idx["agents"]
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
        are_existing_agents = jnp.array(
            [i < self.n_agents_initial for i in range(self.n_agents_max)],
            dtype=jnp.bool_,
        )
        positions_agents = jax.random.randint(
            key=subkey,
            shape=(self.n_agents_max, 2),
            minval=0,
            maxval=max(H, W),
        )
        positions_agents %= jnp.array([H, W])
        map = map.at[
            positions_agents[:, 0],
            positions_agents[:, 1],
            idx_agents,
        ].add(are_existing_agents)

        key_random, subkey = jax.random.split(key_random)
        orientation_agents = jax.random.randint(
            key=subkey,
            shape=(self.n_agents_max,),
            minval=0,
            maxval=4,
        )
        energy_agents = jnp.ones(self.n_agents_max) * self.energy_initial
        age_agents = jnp.zeros(self.n_agents_max)
        appearance_agents = (
            jnp.zeros((self.n_agents_max, self.config["dim_appearance"]))
            .at[: self.n_agents_initial, :]
            .set(1)
        )

        # Initialize the state
        agents = AgentGriworld(
            positions_agents=positions_agents,
            orientation_agents=orientation_agents,
            are_existing_agents=are_existing_agents,
            energy_agents=energy_agents,
            age_agents=age_agents,
            appearance_agents=appearance_agents,
        )

        # Initialize the metrics
        self.aggregators_lifespan: List[Aggregator] = []
        list_metrics_lifespan: List[PyTreeNode] = []
        for config_agg in self.config["metrics"]["aggregators_lifespan"]:
            agg: Aggregator = instantiate_class(**config_agg)
            self.aggregators_lifespan.append(agg)
            list_metrics_lifespan.append(agg.get_initial_metrics())

        self.aggregators_population: List[Aggregator] = []
        list_metrics_population: List[PyTreeNode] = []
        for config_agg in self.config["metrics"]["aggregators_population"]:
            agg: Aggregator = instantiate_class(**config_agg)
            self.aggregators_population.append(agg)
            list_metrics_population.append(agg.get_initial_metrics())

        # Initialize the video memory
        video = jnp.zeros((self.n_steps_per_video, H, W, 3))

        # Initialize ecological informations
        are_newborns_agents = jnp.zeros(self.n_agents_max, dtype=jnp.bool_)
        are_dead_agents = jnp.zeros(self.n_agents_max, dtype=jnp.bool_)
        indexes_parents_agents = jnp.full((self.n_agents_max, 1), self.fill_value)
        eco_information = EcoInformation(
            are_newborns_agents=are_newborns_agents,
            indexes_parents=indexes_parents_agents,
            are_just_dead_agents=are_dead_agents,
        )

        # Initialize the state
        state = StateEnvGridworld(
            timestep=0,
            map=map,
            latitude_sun=latitude_sun,
            agents=agents,
            metrics_lifespan=list_metrics_lifespan,
            metrics_population=list_metrics_population,
            video=video,
        )

        # Return the information required by the agents
        observations_agents, _ = self.get_observations_agents(state=state)
        return (
            state,
            observations_agents,
            eco_information,
            jnp.array(False),
            {},
        )

    def step(
        self,
        state: StateEnvGridworld,
        actions: jnp.ndarray,
        key_random: jnp.ndarray,
        state_species: Optional[StateSpecies] = None,
    ) -> Tuple[
        StateEnvGridworld,
        ObservationAgent,
        EcoInformation,
        bool,
        Dict[str, Any],
    ]:
        """A step of the environment. This function will update the environment according to the actions of the agents.

        Args:
            state (StateEnvGridworld): the state of the environment at timestep t
            actions (jnp.ndarray): the actions of the agents reacting to the environment at timestep t
            key_random (jnp.ndarray): the random key used to generate random numbers

        Returns:
            state_new (StateEnvGridworld): the new state of the environment at timestep t+1
            observations_agents (ObservationAgent): the observations of the agents at timestep t+1
            eco_information (EcoInformation): the ecological information of the environment regarding what happened at t. It should contain the following:
                1) are_newborns_agents (jnp.ndarray): a boolean array indicating which agents are newborns at this step
                2) indexes_parents_agents (jnp.ndarray): an array indicating the indexes of the parents of the newborns at this step
                3) are_dead_agents (jnp.ndarray): a boolean array indicating which agents are dead at this step (i.e. they were alive at t but not at t+1)
                    Note that an agent index could see its are_dead_agents value be False while its are_newborns_agents value is True, if the agent die and another agent is born at the same index
            done (bool): whether the environment is done
            info (Dict[str, Any]): additional information about the environment at timestep t
        """

        H, W, C = state.map.shape
        idx_agents = self.dict_name_channel_to_idx["agents"]

        # Helper func to update agent map
        def update_agent_map(state: StateEnvGridworld) -> StateEnvGridworld:
            map_agents_new = (
                jnp.zeros((H, W))
                .at[
                    state.agents.positions_agents[:, 0],
                    state.agents.positions_agents[:, 1],
                ]
                .add(state.agents.are_existing_agents)
            )
            return state.replace(map=state.map.at[:, :, idx_agents].set(map_agents_new))

        # Initialize the measures dictionnary. This will be used to store the measures of the environment at this step.
        dict_measures_all: Dict[str, jnp.ndarray] = {}
        t = state.timestep

        # ============ (1) Agents interaction with the environment ============
        # Apply the actions of the agents on the environment
        key_random, subkey = jax.random.split(key_random)
        state_new, dict_measures = self.step_action_agents(
            state=state, actions=actions, key_random=subkey
        )
        dict_measures_all.update(dict_measures)
        state_new = update_agent_map(state_new)

        # ============ (2) Agents reproduce ============
        key_random, subkey = jax.random.split(key_random)
        state_new, are_newborns_agents, indexes_parents_agents, dict_measures = (
            self.step_reproduce_agents(
                state=state_new, actions=actions, key_random=subkey
            )
        )
        dict_measures_all.update(dict_measures)
        state_new = update_agent_map(state_new)

        # ============ (3) Extract the observations of the agents (and some updates) ============
        # update agent ages in map
        norm_factor = jnp.maximum(1, state.map[..., idx_agents])
        map_ages_new = (
            jnp.zeros((H, W))
            .at[
                state.agents.positions_agents[:, 0],
                state.agents.positions_agents[:, 1],
            ]
            .add(state.agents.age_agents * state.agents.are_existing_agents)
            / norm_factor
        )

        # Recreate the map of appearances
        map_appearances_new = jnp.zeros((H, W, self.config["dim_appearance"])).at[
            state.agents.positions_agents[:, 0],
            state.agents.positions_agents[:, 1],
            :,
        ].add(
            state.agents.appearance_agents * state.agents.are_existing_agents[:, None]
        ) / norm_factor.reshape(
            (H, W, 1)
        )

        # Update the state
        map_new = state_new.map.at[:, :, idx_agents + 1].set(map_ages_new)
        map_new = map_new.at[:, :, idx_agents + 2 :].set(map_appearances_new)
        state_new: StateEnvGridworld = state_new.replace(
            map=map_new,
            timestep=t + 1,
            agents=state_new.agents.replace(age_agents=state_new.agents.age_agents + 1),
        )

        # Extract the observations of the agents
        observations_agents, dict_measures = self.get_observations_agents(
            state=state_new
        )
        dict_measures_all.update(dict_measures)

        # ============ (4) Get the ecological information ============
        are_just_dead_agents = state_new.agents.are_existing_agents & (
            ~state_new.agents.are_existing_agents
            | (state_new.agents.age_agents < state_new.agents.age_agents)
        )
        eco_information = EcoInformation(
            are_newborns_agents=are_newborns_agents,
            indexes_parents=indexes_parents_agents,
            are_just_dead_agents=are_just_dead_agents,
        )

        # ============ (5) Check if the environment is done ============
        if self.is_terminal:
            done = ~jnp.any(state_new.agents.are_existing_agents)
        else:
            done = False

        # ============ (6) Compute the metrics ============
        # Compute some measures
        dict_measures = self.compute_measures(
            state=state, actions=actions, state_new=state_new, key_random=subkey, state_species=state_species
        )
        dict_measures_all.update(dict_measures)

        # Set the measures to NaN for the agents that are not existing
        for name_measure, measures in dict_measures_all.items():
            if name_measure not in self.config["metrics"]["measures"]["environmental"]:
                dict_measures_all[name_measure] = jnp.where(
                    state_new.agents.are_existing_agents,
                    measures,
                    jnp.nan,
                )

        # Update and compute the metrics
        state_new, dict_metrics = self.compute_metrics(
            state=state, state_new=state_new, dict_measures=dict_measures_all
        )
        info = {"metrics": dict_metrics}

        # ============ (7) Manage the video ============
        # Reset the video to empty if t = 0 mod n_steps_per_video
        video = jax.lax.cond(
            t % self.n_steps_per_video == 0,
            lambda _: jnp.zeros((self.n_steps_per_video, H, W, 3)),
            lambda _: state_new.video,
            operand=None,
        )
        # Add the new frame to the video
        rgb_map = self.get_RGB_map(images=state_new.map)
        video = state_new.video.at[t % self.n_steps_per_video].set(rgb_map)
        # Update the state
        state_new = state_new.replace(video=video)

        # Return the new state and observations
        return (
            state_new,
            observations_agents,
            eco_information,
            done,
            info,
        )

    def get_observation_space(self) -> DictSpace:
        return self.observation_space

    def get_action_space(self) -> DiscreteSpace:
        return DiscreteSpace(n=self.n_actions)

    def render(self, state: StateEnvGridworld) -> None:
        """The rendering function of the environment. It saves the RGB map of the environment as a video."""
        if not self.cfg_video["do_video"]:
            return
        t = state.timestep
        if t < self.n_steps_per_video:
            return  # Not enough frames to render a video

        tqdm.write(f"Rendering video at timestep {t}...")
        video_writer = VideoRecorder(
            filename=f"{self.dir_videos}/video_t{t}.mp4",
            fps=self.fps_video,
        )
        for t_ in range(self.n_steps_per_video):
            image = state.video[t_]
            image = self.upscale_image(image)
            video_writer.add(image)
        video_writer.close()

    # ================== Helper functions ==================
    @partial(jax.jit, static_argnums=(0,))
    def get_RGB_map(self, images: jnp.ndarray) -> jnp.ndarray:
        """Get the RGB map by applying a color to each channel of a list of grey images and blend them together

        Args:
            images (np.ndarray): the array of grey images, of shape (height, width, channels)
            dict_idx_channel_to_color_tag (Dict[int, tuple]): a mapping from channel index to color tag.
                A color tag is a tuple of 3 floats between 0 and 1

        Returns:
            np.ndarray: the blended image, of shape (height, width, 3), with the color applied to each channel,
                with pixel values between 0 and 1
        """
        # Initialize an empty array to store the blended image
        assert (
            self.color_tag_background in DICT_COLOR_TAG_TO_RGB
        ), f"Unknown color tag: {self.color_tag_background}"
        background = jnp.array(
            DICT_COLOR_TAG_TO_RGB[self.color_tag_background], dtype=jnp.float32
        )

        blended_image = background * jnp.ones(
            images.shape[:2] + (3,), dtype=jnp.float32
        )

        # Iterate over each channel and apply the corresponding color.
        # For each channel, we set the color at each tile to the channel colour
        # with an intensity proportional to the number of entities (of that channel)
        # in the tile, with nonzero intensities scaled to be between 0.3 and 1
        for channel_idx, color_tag in self.dict_idx_channel_to_color_tag.items():
            channel_name = self.list_names_channels[channel_idx]
            if channel_name not in self.dict_name_channel_to_color_tag:
                continue

            delta = jnp.array(
                DICT_COLOR_TAG_TO_RGB[color_tag], dtype=jnp.float32
            ) - jnp.array([1, 1, 1], dtype=jnp.float32)
            img = images[:, :, channel_idx][:, :, None]
            intensity = jnp.where(img > 0, img / jnp.maximum(1, jnp.max(img)), 0)
            intensity = jnp.where(intensity > 0, (intensity * 0.7) + 0.3, 0)
            blended_image += delta * intensity

        # Clip all rgb values to be between 0 and 1
        return jnp.clip(blended_image, 0, 1)

    def upscale_image(self, image: jnp.ndarray) -> jnp.ndarray:
        """Upscale an image to a maximum size while keeping the aspect ratio.

        Args:
            image (jnp.ndarray): the image to scale, of shape (H, W, C)

        Returns:
            jnp.ndarray: the scaled image, of shape (H', W', C), with H' <= self.height_max_video and W' <= self.width_max_video
        """
        H, W, C = image.shape
        upscale_factor = min(self.height_max_video / H, self.width_max_video / W)
        upscale_factor = int(upscale_factor)
        assert upscale_factor >= 1, "The upscale factor must be at least 1"
        image_upscaled = jax.image.resize(
            image,
            shape=(H * upscale_factor, W * upscale_factor, C),
            method="nearest",
        )
        return image_upscaled

    def step_grow_plants(
        self, state: StateEnvGridworld, key_random: jnp.ndarray
    ) -> jnp.ndarray:
        """Modify the state of the environment by growing the plants."""
        idx_sun = self.dict_name_channel_to_idx["sun"]
        idx_plants = self.dict_name_channel_to_idx["plants"]
        map_plants = state.map[:, :, self.dict_name_channel_to_idx["plants"]]
        # map_n_plant_in_radius_plant_reproduction = convolve2d(
        #     map_plants,
        #     self.kernel_plant_reproduction,
        #     mode="same",
        # )
        # map_n_plant_in_radius_plant_asphyxia = convolve2d(
        #     map_plants,
        #     self.kernel_plant_asphyxia,
        #     mode="same",
        # )
        map_sun = state.map[:, :, idx_sun]
        logits_plants = (
            self.logit_p_base_plant_growth * (1 - map_plants)
            + (1 - self.logit_p_base_plant_death) * map_plants
            # + self.factor_sun_effect * map_sun
            # + self.factor_plant_reproduction * map_n_plant_in_radius_plant_reproduction
            # - self.factor_plant_asphyxia * map_n_plant_in_radius_plant_asphyxia
        )
        logits_plants = jnp.clip(logits_plants, -10, 10)
        map_plants_probs = sigmoid(x=logits_plants)
        key_random, subkey = jax.random.split(key_random)
        map_plants = jax.random.bernoulli(
            key=subkey,
            p=map_plants_probs,
            shape=map_plants.shape,
        )
        return state.replace(map=state.map.at[:, :, idx_plants].set(map_plants))

    def step_update_sun(
        self, state: StateEnvGridworld, key_random: jnp.ndarray
    ) -> StateEnvGridworld:
        """Modify the state of the environment by updating the sun.
        The method of updating the sun is defined by the attribute self.method_sun.
        """
        # Update the latitude of the sun depending on the method
        idx_sun = self.dict_name_channel_to_idx["sun"]
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
            latitude_sun = H // 2 + H * state.timestep // self.period_sun
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

    def get_facing_pos(self, position, orientation) -> jnp.ndarray:
        angle = orientation * jnp.pi / 2
        d_pos = jnp.array([jnp.cos(angle), -jnp.sin(angle)]).astype(jnp.int32)
        return (position + d_pos) % jnp.array([self.height, self.width])

    def compute_new_position_and_orientation(
        self,
        key_random: jnp.ndarray,
        curr_pos: jnp.ndarray,
        curr_ori: jnp.ndarray,
        action: jnp.ndarray,
        move_success_prob: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        condlist: List[jnp.ndarray] = []
        choicelist_position: List[jnp.ndarray] = []
        choicelist_orientation: List[jnp.ndarray] = []

        for name_action_move in ["forward", "backward", "left", "right"]:
            # Check if the action is a move action
            if name_action_move not in self.list_actions:
                continue

            # Add the action to the list of possible actions related to moving
            condlist.append(action == self.action_to_idx[name_action_move])

            # Add the new orientation to the list of possible orientations
            direction = ["forward", "left", "backward", "right"].index(name_action_move)
            new_ori = (curr_ori + direction) % 4
            choicelist_orientation.append(new_ori)

            # Add the new position to the list of possible positions
            new_pos = self.get_facing_pos(curr_pos, new_ori)
            choicelist_position.append(new_pos)

        # Select the new position and orientation of the agent based on the action
        new_pos = jnp.select(
            condlist=condlist,
            choicelist=choicelist_position,
            default=curr_pos,
        )

        new_ori = jnp.select(
            condlist=condlist,
            choicelist=choicelist_orientation,
            default=curr_ori,
        )

        # Decide if the move is successful
        success = jax.random.bernoulli(key=key_random, p=move_success_prob)
        new_pos = jnp.where(success, new_pos, curr_pos)

        return new_pos, new_ori

    def move_agents_allow_multiple_occupancy(
        self, key_random: jnp.ndarray, state: StateEnvGridworld, actions: jnp.ndarray
    ):
        keys = jax.random.split(key_random, num=state.agents.positions_agents.shape[0])
        success_probs = jnp.minimum(
            self.move_prob_initial
            + (self.move_prob_gradient * state.agents.age_agents),
            1,
        )
        return jax.vmap(
            self.compute_new_position_and_orientation, in_axes=(0, 0, 0, 0, 0)
        )(
            keys,
            state.agents.positions_agents,
            state.agents.orientation_agents,
            actions,
            success_probs,
        )

    def move_agents_enforce_single_occupancy(
        self, key_random: jnp.ndarray, state: StateEnvGridworld, actions: jnp.ndarray
    ):
        num_agents = state.agents.positions_agents.shape[0]
        keys = jax.random.split(key_random, num=num_agents)
        success_probs = jnp.minimum(
            self.move_prob_initial
            + (self.move_prob_gradient * state.agents.age_agents),
            1,
        )

        def process_single_agent(agent_map, agent_idx):
            curr_pos = state.agents.positions_agents[agent_idx]
            curr_ori = state.agents.orientation_agents[agent_idx]
            new_pos, new_ori = self.compute_new_position_and_orientation(
                keys[agent_idx],
                curr_pos,
                curr_ori,
                actions[agent_idx],
                success_probs[agent_idx],
            )

            # check whether new position is occupied - if so, agent stays in place
            new_pos = jnp.where(
                agent_map[new_pos[0], new_pos[1]] > 0,
                curr_pos,
                new_pos,
            )

            agent_map = agent_map.at[new_pos[0], new_pos[1]].set(1)
            return agent_map, jnp.array([new_pos[0], new_pos[1], new_ori])

        agent_map = state.map[..., self.dict_name_channel_to_idx["agents"]]
        agent_map, outputs = jax.lax.scan(
            process_single_agent, agent_map, jnp.arange(num_agents)
        )

        return (
            outputs[:, :2],
            outputs[:, 2],
        )

    def step_action_agents(
        self,
        state: StateEnvGridworld,
        actions: jnp.ndarray,
        key_random: jnp.ndarray,
    ) -> Tuple[StateEnvGridworld, Dict[str, jnp.ndarray]]:
        """Modify the state of the environment by applying the actions of the agents."""
        H, W, C = state.map.shape
        idx_plants = self.dict_name_channel_to_idx["plants"]
        idx_agents = self.dict_name_channel_to_idx["agents"]
        map_plants = state.map[..., idx_plants]
        map_agents = state.map[..., idx_agents]
        dict_measures: Dict[str, jnp.ndarray] = {}

        # ====== Compute the new positions and orientations of all the agents ======
        move_func = (
            self.move_agents_allow_multiple_occupancy
            if self.allow_multiple_agents_per_tile
            else self.move_agents_enforce_single_occupancy
        )
        positions_agents_new, orientation_agents_new = move_func(
            key_random, state, actions
        )

        # ====== Perform the eating action of the agents ======
        if "eat" in self.list_actions:
            are_agents_eating = state.agents.are_existing_agents & (
                actions == self.action_to_idx["eat"]
            )
            map_agents_try_eating = (
                jnp.zeros_like(map_agents)
                .at[positions_agents_new[:, 0], positions_agents_new[:, 1]]
                .add(are_agents_eating)
            )  # map of the number of (existing) agents trying to eat at each cell

            map_food_energy_bonus_available_per_agent = (
                self.energy_food * map_plants / jnp.maximum(1, map_agents_try_eating)
            )  # map of the energy available at each cell per (existing) agent trying to eat
            food_energy_bonus = (
                map_food_energy_bonus_available_per_agent[
                    positions_agents_new[:, 0], positions_agents_new[:, 1]
                ]
                * are_agents_eating
            )

            if "amount_food_eaten" in self.names_measures:
                dict_measures["amount_food_eaten"] = food_energy_bonus
            energy_agents_new = state.agents.energy_agents + food_energy_bonus

            if "eat_success_rate" in self.names_measures:
                dict_measures["eat_success_rate"] = jnp.sign(
                    food_energy_bonus
                ).sum() / jnp.maximum(1, are_agents_eating.sum())

            # Remove plants that have been eaten
            map_plants -= map_agents_try_eating * map_plants
            map_plants = jnp.clip(map_plants, 0, 1)

        # ====== Handle any energy transfer actions ======
        if "transfer" in self.list_actions:
            # Check if any agents are transferring energy
            are_agents_transferring = state.agents.are_existing_agents & (
                actions == self.action_to_idx["transfer"]
            )

            def per_agent_helper_fn(i):
                target_pos = self.get_facing_pos(
                    state.agents.positions_agents[i], state.agents.orientation_agents[i]
                )
                are_receiving = (
                    (state.agents.positions_agents == target_pos)
                    .all(axis=1)
                    .astype(jnp.int32)
                )
                is_transfer = are_agents_transferring[i] & jnp.any(are_receiving)

                loss = is_transfer * self.energy_transfer_loss
                gain = (
                    is_transfer
                    * self.energy_transfer_gain
                    / jnp.maximum(1, jnp.sum(are_receiving))
                )

                return (
                    jnp.zeros_like(state.agents.energy_agents).at[i].add(-loss)
                    + (gain * are_receiving).astype(jnp.float32),
                    is_transfer,
                )

            transfer_delta_energy, is_transfer = jax.vmap(
                per_agent_helper_fn, in_axes=0
            )(jnp.arange(state.agents.energy_agents.size))
            energy_agents_new += transfer_delta_energy.sum(axis=0)

            # log some metrics
            if "net_energy_transfer_per_capita" in self.names_measures:
                dict_measures["net_energy_transfer_per_capita"] = (
                    transfer_delta_energy.sum() / state.agents.are_existing_agents.sum()
                )
            if "transfer_success_rate" in self.names_measures:
                dict_measures["transfer_success_rate"] = jnp.sum(is_transfer) / jnp.sum(
                    are_agents_transferring
                )

        # ====== Update the physical status of the agents ======
        idle_agents = state.agents.are_existing_agents & (
            actions == self.action_to_idx["idle"]
        )
        not_idle_agents = state.agents.are_existing_agents & ~idle_agents & ~are_agents_transferring
        energy_agents_new = (
            energy_agents_new
            - self.energy_loss_idle * idle_agents
            - self.energy_loss_action * not_idle_agents
        )
        energy_agents_new = jnp.clip(energy_agents_new, 0, self.energy_max)

        are_existing_agents_new = (
            (energy_agents_new > self.energy_thr_death)
            & state.agents.are_existing_agents
            & (state.agents.age_agents < self.age_max)
        )
        if "life_expectancy" in self.names_measures:
            just_died = state.agents.are_existing_agents & ~are_existing_agents_new
            le = jnp.sum(just_died * state.agents.age_agents) / jnp.maximum(
                1, jnp.sum(just_died)
            )
            dict_measures["life_expectancy"] = le

        appearance_agents_new = (
            state.agents.appearance_agents * are_existing_agents_new[:, None]
        )

        # Update the state
        agents_new = state.agents.replace(
            positions_agents=positions_agents_new,
            orientation_agents=orientation_agents_new,
            energy_agents=energy_agents_new,
            are_existing_agents=are_existing_agents_new,
            appearance_agents=appearance_agents_new,
        )
        state = state.replace(
            map=state.map.at[:, :, idx_plants].set(map_plants),
            agents=agents_new,
        )

        # Update the sun
        key_random, subkey = jax.random.split(key_random)
        state = self.step_update_sun(state=state, key_random=subkey)
        # Grow plants
        key_random, subkey = jax.random.split(key_random)
        state = self.step_grow_plants(state=state, key_random=subkey)

        # Return the new state, as well as some metrics
        return state, dict_measures

    def step_reproduce_agents(
        self,
        state: StateEnvGridworld,
        actions: jnp.ndarray,
        key_random: jnp.ndarray,
    ) -> Tuple[StateEnvGridworld, jnp.ndarray, jnp.ndarray]:
        """Reproduce the agents in the environment."""
        dict_measures = {}

        # Detect which agents are trying to reproduce
        are_existing_agents = state.agents.are_existing_agents
        are_agents_trying_reprod = (
            (state.agents.energy_agents > self.energy_req_reprod)
            & (state.agents.age_agents >= 25)
            & are_existing_agents
        )
        if "reproduce" in self.list_actions:
            trying_reprod_action = actions == self.action_to_idx["reproduce"]
            are_agents_trying_reprod = are_agents_trying_reprod & trying_reprod_action
            if "reproduce_success_rate" in self.names_measures:
                dict_measures["reproduce_success_rate"] = jnp.sum(
                    are_agents_trying_reprod
                ) / jnp.sum(trying_reprod_action)

        # # For test
        # are_existing_agents = jnp.array([False, False, False, False, True, True, True, True, True, False])
        # are_agents_trying_reprod = jnp.array([False, False, False, False, False, False, False, False, False, False])

        # Compute the number of newborns. If there are more agents trying to reproduce than there are ghost agents, only the first n_ghost_agents agents will be able to reproduce.
        n_agents_trying_reprod = jnp.sum(are_agents_trying_reprod)
        n_ghost_agents = jnp.sum(~are_existing_agents)
        n_newborns = jnp.minimum(n_agents_trying_reprod, n_ghost_agents)

        # Compute which agents are actually reproducing
        try_reprod_mask = are_agents_trying_reprod.astype(
            jnp.int32
        )  # 1_(agent i tries to reproduce) for i
        cumsum_repro_attempts = jnp.cumsum(
            try_reprod_mask
        )  # number of agents that tried to reproduce before agent i
        are_agents_reproducing = (
            cumsum_repro_attempts <= n_newborns
        ) & are_agents_trying_reprod  # whether i tried to reproduce and is allowed to reproduce

        if "amount_children" in self.names_measures:
            dict_measures["amount_children"] = are_agents_reproducing

        # Get the indices of the ghost agents. To have constant (n_max_agents,) shape, we fill the remaining indices with the value self.n_agents_max (which will have no effect as an index of (n_agents_max,) array)
        indices_ghost_agents_FILLED = jnp.where(
            ~are_existing_agents,
            size=self.n_agents_max,
            fill_value=self.fill_value,
        )[
            0
        ]  # placeholder_indices = [i1, i2, ..., i(n_ghost_agents), f, f, ..., f] of shape (n_max_agents,)

        # Get the indices of the ghost agents that will become newborns and define the newborns
        indices_newborn_agents_FILLED = jnp.where(
            jnp.arange(self.n_agents_max) < n_newborns,
            indices_ghost_agents_FILLED,
            self.n_agents_max,
        )  # placeholder_indices = [i1, i2, ..., i(n_newborns), f, f, ..., f] of shape (n_max_agents,), with n_newborns <= n_ghost_agents

        are_newborns_agents = (
            jnp.zeros(self.n_agents_max, dtype=jnp.bool_)
            .at[indices_newborn_agents_FILLED]
            .set(True)
        )  # whether agent i is a newborn

        # Get the indices of are_reproducing agents
        indices_had_reproduced_FILLED = jnp.where(
            are_agents_reproducing,
            size=self.n_agents_max,
            fill_value=self.fill_value,
        )[0]

        agents_parents = jnp.full(
            shape=(self.n_agents_max, 1), fill_value=self.fill_value, dtype=jnp.int32
        )
        agents_parents = agents_parents.at[indices_newborn_agents_FILLED].set(
            indices_had_reproduced_FILLED[:, None]
        )

        # Decrease the energy of the agents that are reproducing
        energy_agents_new = (
            state.agents.energy_agents
            - are_agents_reproducing * self.energy_cost_reprod
        )

        # Initialize the newborn agents
        are_existing_agents_new = are_existing_agents | are_newborns_agents
        energy_agents_new = energy_agents_new.at[indices_newborn_agents_FILLED].set(
            self.energy_initial
        )
        age_agents_new = state.agents.age_agents.at[indices_newborn_agents_FILLED].set(
            0
        )

        # Initialize the newborn agents' position
        H, W = self.height, self.width
        dummy_val = 2 * jnp.maximum(H, W)

        def place_newborn_in_empty_tile(empty_locs, newborn_idx):
            # find the nearest free square to the newborn's parent
            # and place them there
            parent_loc = state.agents.positions_agents[agents_parents[newborn_idx]]
            closest_idx = jnp.argmin(jnp.sum(jnp.abs(empty_locs - parent_loc), axis=1))
            closest_loc = empty_locs[closest_idx]
            empty_locs = empty_locs.at[closest_idx].set([dummy_val, dummy_val])
            return empty_locs, closest_loc

        if self.allow_multiple_agents_per_tile:
            newborn_positions = state.agents.positions_agents[
                indices_had_reproduced_FILLED
            ]
        else:
            agent_map = state.map[..., self.dict_name_channel_to_idx["agents"]]
            empty_locs = jnp.stack(
                jnp.where(agent_map == 0, size=H * W, fill_value=dummy_val), axis=1
            )
            _, newborn_positions = jax.lax.scan(
                place_newborn_in_empty_tile, empty_locs, indices_newborn_agents_FILLED
            )

        positions_agents_new = state.agents.positions_agents.at[
            indices_newborn_agents_FILLED
        ].set(newborn_positions)

        # Initialize the newborn agents' appearances
        key_random, subkey = jax.random.split(key_random)
        noise_appearances = (
            jax.random.normal(
                key=subkey,
                shape=(self.n_agents_max, self.config["dim_appearance"]),
            )
            * 0.001
        )
        appearance_agents_new = state.agents.appearance_agents.at[
            indices_newborn_agents_FILLED
        ].set(
            state.agents.appearance_agents[indices_had_reproduced_FILLED]
            + noise_appearances
        )

        # Update the state
        agents_new = state.agents.replace(
            energy_agents=energy_agents_new,
            are_existing_agents=are_existing_agents_new,
            age_agents=age_agents_new,
            positions_agents=positions_agents_new,
            appearance_agents=appearance_agents_new,
        )
        state = state.replace(agents=agents_new)

        return (
            state,
            are_newborns_agents,
            agents_parents,
            dict_measures,
        )

    def get_observations_agents(
        self, state: StateEnvGridworld
    ) -> Tuple[ObservationAgent, Dict[str, jnp.ndarray]]:
        """Extract the observations of the agents from the state of the environment.

        Args:
            state (StateEnvGridworld): the state of the environment

        Returns:
            observation_agents (ObservationAgent): the observations of the agents
            dict_measures (Dict[str, jnp.ndarray]): a dictionary of the measures of the environment
        """

        def get_single_agent_visual_field(
            agents: AgentGriworld,
        ) -> jnp.ndarray:
            """Get the visual field of a single agent.

            Args:
                agent_state (StateAgentGriworld): the state of the agent

            Returns:
                jnp.ndarray: the visual field of the agent, of shape (2 * self.vision_radius + 1, 2 * self.vision_radius + 1, ?)
            """
            H, W, C_map = state.map.shape

            # Construct the map of the visual field of the agent
            map_vis_field = state.map[:, :, self.list_indexes_channels_visual_field]  # (H, W, C_map)
            age_idx = self.dict_name_channel_to_idx["agent_ages"]
            map_vis_field = map_vis_field.at[:, :, age_idx].set(
                map_vis_field[:, :, age_idx] / self.age_max
            )

            # Get the visual field of the agent
            visual_field_x = agents.positions_agents[0] + self.grid_indexes_vision_x
            visual_field_y = agents.positions_agents[1] + self.grid_indexes_vision_y
            vis_field = map_vis_field[
                visual_field_x % H,
                visual_field_y % W,
            ]  # (2 * self.vision_radius + 1, 2 * self.vision_radius + 1, C_map)

            # Rotate the visual field according to the orientation of the agent
            vis_field = jnp.select(
                [
                    agents.orientation_agents == 0,
                    agents.orientation_agents == 1,
                    agents.orientation_agents == 2,
                    agents.orientation_agents == 3,
                ],
                [
                    vis_field,
                    jnp.rot90(vis_field, k=1, axes=(0, 1)),
                    jnp.rot90(vis_field, k=2, axes=(0, 1)),
                    jnp.rot90(vis_field, k=3, axes=(0, 1)),
                ],
            )

            # Return the visual field of the agent
            return vis_field

        # Create the observation of the agents
        dict_observations: Dict[str, jnp.ndarray] = {}
        if "energy" in self.list_observations:
            dict_observations["energy"] = state.agents.energy_agents / self.energy_max
        if "age" in self.list_observations:
            dict_observations["age"] = state.agents.age_agents / self.age_max
        if "visual_field" in self.list_observations:
            dict_observations["visual_field"] = jax.vmap(get_single_agent_visual_field)(
                state.agents
            )

        # observations = self.ObservationAgentGridworld(
        #     **{
        #         name_obs: dict_observations[name_obs]
        #         for name_obs in self.list_observations
        #     }
        # )

        dict_measures = {}

        # print(f"Map : {state.map[..., 0]}")
        # print(f"Agents positions : {state.positions_agents}")
        # print(f"Agents orientations : {state.orientation_agents}")
        # print(f"First agent obs : {observations.visual_field[0, ..., 2]}, shape : {observations.visual_field[0, ...].shape}")
        # raise ValueError("Stop")

        return dict_observations, dict_measures

    def compute_measures(
        self,
        state: StateEnvGridworld,
        actions: jnp.ndarray,
        state_new: StateEnvGridworld,
        key_random: jnp.ndarray,
        state_species: Optional[StateSpecies] = None,
    ) -> Dict[str, jnp.ndarray]:
        """Get the measures of the environment.

        Args:
            state (StateEnvGridworld): the state of the environment
            actions (jnp.ndarray): the actions of the agents
            state_new (StateEnvGridworld): the new state of the environment
            key_random (jnp.ndarray): the random key
            state_species (Optional[StateSpecies]): the state of the species

        Returns:
            Dict[str, jnp.ndarray]: a dictionary of the measures of the environment
        """
        dict_measures = {}
        idx_plants = self.dict_name_channel_to_idx["plants"]
        idx_agents = self.dict_name_channel_to_idx["agents"]
        for name_measure in self.names_measures:
            # Environment measures
            if name_measure == "n_agents":
                dict_measures["n_agents"] = jnp.sum(state.agents.are_existing_agents)
            elif name_measure == "n_plants":
                dict_measures["n_plants"] = jnp.sum(state.map[..., idx_plants])
            elif name_measure == "group_size":
                group_sizes = compute_group_sizes(state.map[..., idx_agents])
                dict_measures["average_group_size"] = group_sizes.mean()
                dict_measures["max_group_size"] = group_sizes.max()
                continue
            # Immediate measures
            elif name_measure.startswith("do_action_"):
                str_action = name_measure[len("do_action_") :]
                if str_action in self.list_actions:
                    dict_measures[name_measure] = (
                        actions == self.action_to_idx[str_action]
                    ).astype(jnp.float32)
            # State measures
            elif name_measure == "energy":
                dict_measures[name_measure] = state.agents.energy_agents
            elif name_measure == "age":
                dict_measures[name_measure] = state.agents.age_agents
            elif name_measure == "x":
                dict_measures[name_measure] = state.agents.positions_agents[:, 0]
            elif name_measure == "y":
                dict_measures[name_measure] = state.agents.positions_agents[:, 1]
            elif name_measure == "appearance":
                for i in range(self.config["dim_appearance"]):
                    dict_measures[f"appearance_{i}"] = state.agents.appearance_agents[
                        :, i
                    ]
            # Behavior measures (requires state_species)
            elif name_measure in self.config["metrics"]["measures"]["behavior"]:
                assert isinstance(self.agent_species, AgentSpecies), f"For behavior measure, you need to give an agent species as attribute of the env after both creation : env.agent_species = agent_species"
                dict_measures.update(self.compute_behavior_measure(
                    state_species=state_species,
                    key_random=key_random,
                    name_measure=name_measure,
                ))
            else:
                pass  # Pass this measure as it may be computed in other parts of the code

        # Return the dictionary of measures
        return dict_measures

    def compute_metrics(
        self,
        state: StateEnvGridworld,
        state_new: StateEnvGridworld,
        dict_measures: Dict[str, jnp.ndarray],
    ):

        # Set the measures to NaN for the agents that are not existing
        for name_measure, measures in dict_measures.items():
            if name_measure not in self.config["metrics"]["measures"]["environmental"]:
                dict_measures[name_measure] = jnp.where(
                    state_new.agents.are_existing_agents,
                    measures,
                    jnp.nan,
                )

        # Aggregate the measures over the lifespan
        are_just_dead_agents = state_new.agents.are_existing_agents & (
            ~state_new.agents.are_existing_agents
            | (state_new.agents.age_agents < state_new.agents.age_agents)
        )

        dict_metrics_lifespan = {}
        new_list_metrics_lifespan = []
        for agg, metrics in zip(self.aggregators_lifespan, state.metrics_lifespan):
            new_metrics = agg.update_metrics(
                metrics=metrics,
                dict_measures=dict_measures,
                are_alive=state_new.agents.are_existing_agents,
                are_just_dead=are_just_dead_agents,
                ages=state_new.agents.age_agents,
            )
            dict_metrics_lifespan.update(agg.get_dict_metrics(new_metrics))
            new_list_metrics_lifespan.append(new_metrics)
        state_new_new = state_new.replace(metrics_lifespan=new_list_metrics_lifespan)

        # Aggregate the measures over the population
        dict_metrics_population = {}
        new_list_metrics_population = []
        for agg, metrics in zip(self.aggregators_population, state.metrics_population):
            new_metrics = agg.update_metrics(
                metrics=metrics,
                dict_measures=dict_measures,
                are_alive=state_new.agents.are_existing_agents,
                are_just_dead=are_just_dead_agents,
                ages=state_new.agents.age_agents,
            )
            dict_metrics_population.update(agg.get_dict_metrics(new_metrics))
            new_list_metrics_population.append(new_metrics)
        state_new_new = state_new_new.replace(
            metrics_population=new_list_metrics_population
        )

        # Get the final metrics
        dict_metrics = {
            **dict_measures,
            **dict_metrics_lifespan,
            **dict_metrics_population,
        }

        # Arrange metrics in right format
        for name_metric in list(dict_metrics.keys()):
            *names, name_measure = name_metric.split("/")
            if len(names) == 0:
                name_metric_new = name_measure
            else:
                name_metric_new = f"{name_measure}/{' '.join(names[::-1])}"
            dict_metrics[name_metric_new] = dict_metrics.pop(name_metric)

        return state_new_new, dict_metrics


# ================== Helper functions ==================


def compute_group_sizes(agent_map: jnp.ndarray) -> float:
    H, W = agent_map.shape
    done = set()

    def dfs(i, j):
        if (i, j) in done:
            return 0
        done.add((i, j))

        if i < 0 or j < 0 or i >= H or j >= W or agent_map[i, j] == 0:
            return 0

        return (
            int(agent_map[i, j])
            + dfs(i + 1, j)
            + dfs(i - 1, j)
            + dfs(i, j + 1)
            + dfs(i, j - 1)
            + dfs(i - 1, j - 1)
            + dfs(i - 1, j + 1)
            + dfs(i + 1, j - 1)
            + dfs(i + 1, j + 1)
        )

    groups = jnp.array(
        [dfs(i, j) for i in range(H) for j in range(W) if agent_map[i, j] > 0]
    )
    return groups[groups > 0]
