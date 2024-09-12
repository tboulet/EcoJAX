# Gridworld EcoJAX environment

from collections import defaultdict
from functools import partial
import os
from time import sleep
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar

import jax
import jax.numpy as jnp
from jax.numpy import ndarray
from matplotlib import pyplot as plt
from matplotlib.pylab import f
import numpy as np
from jax import random
from jax.scipy.signal import convolve2d
from flax.struct import PyTreeNode, dataclass
from jax.debug import breakpoint as jbreakpoint
from tqdm import tqdm

from ecojax.agents.base_agent_species import AgentSpecies
from ecojax.core.eco_info import EcoInformation
from ecojax.environment import EcoEnvironment
from ecojax.metrics.aggregators import Aggregator
from ecojax.spaces import DictSpace, EcojaxSpace, DiscreteSpace, ContinuousSpace
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
    # The energy level of the agents, of shape (n_max_agents,). energy_agents[i] represents the energy level of the i-th agent. This is normalized by energy_max in the observation.
    energy_agents: jnp.ndarray  # (n_max_agents,) in [0, +inf)
    # The age of the agents. This is normalized by age_max in the observation.
    age_agents: jnp.ndarray  # (n_max_agents,) in [0, +inf)
    # The appearance of the agent, encoded as a vector in R^dim_appearance. appearance_agents[i, :] represents the appearance of the i-th agent.
    # The appearance of an agent allows the agents to distinguish their genetic proximity, as agents with similar appearances are more likely to be genetically close.
    # By convention : a non-agent has an appearance of zeros, the common ancestors have an appearance of ones, and m superposed agents have an appearance of their average.
    appearance_agents: jnp.ndarray  # (n_max_agents, dim_appearance) in R
    # A counter for each fruit of the number of timesteps ago it was eaten by the agent. The counter is reset to 0 when the agent eats the fruit.
    # This counter is initialized at -age_max when the agent is born (child curiostity) and is normalized by age_max in the observation
    novelty_hunger: jnp.ndarray  # (n_max_agents, 4) in R+
    # The number of child an agent had in its life.
    n_childrens: jnp.ndarray  # (n_max_agents,) in N


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
    video_agent: jnp.ndarray  # (n_steps_per_video, 2*v+1, 2*v+1, 3) in [0, 1]


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
        self.do_fruits: bool = config["do_fruits"]
        self.list_names_channels: List[str] = [
            "sun",
            "plants",
            "agents",
        ]
        self.list_names_channels += [
            f"appearance_{i}" for i in range(config["dim_appearance"])
        ]
        if self.do_fruits:
            self.list_names_channels += [f"fruits_{i}" for i in range(4)]
        self.dict_name_channel_to_idx: Dict[str, int] = {
            name_channel: idx_channel
            for idx_channel, name_channel in enumerate(self.list_names_channels)
        }
        self.n_channels_map: int = len(self.dict_name_channel_to_idx)
        self.list_indexes_channels_visual_field: List[int] = []
        for name_channel in config["list_channels_visual_field"]:
            assert (
                name_channel in self.dict_name_channel_to_idx
            ), f"Unknown channel: {name_channel} not in {self.dict_name_channel_to_idx}"
            self.list_indexes_channels_visual_field.append(
                self.dict_name_channel_to_idx[name_channel]
            )
        self.dict_name_channel_to_idx_visual_field: Dict[str, int] = {
            name_channel: idx_channel
            for idx_channel, name_channel in enumerate(
                config["list_channels_visual_field"]
            )
        }
        self.n_channels_visual_field: int = len(self.list_indexes_channels_visual_field)
        # Metrics parameters
        self.names_measures: List[str] = sum(
            [names for type_measure, names in config["metrics"]["measures"].items()], []
        )
        self.behavior_measures_on_render: List[str] = config["metrics"][
            "behavior_measures_on_render"
        ]
        # Video parameters
        self.cfg_video = config["metrics"]["config_video"]
        self.do_video: bool = self.cfg_video["do_video"]
        self.do_agent_video: bool = self.cfg_video["do_agent_video"]
        self.n_steps_per_video: int = self.cfg_video["n_steps_per_video"]
        self.t_last_video_rendered: int = -float("inf")
        self.n_steps_min_between_videos = self.cfg_video["n_steps_min_between_videos"]
        self.fps_video: int = self.cfg_video["fps_video"]
        self.dir_videos: str = self.cfg_video["dir_videos"]
        os.makedirs(self.dir_videos, exist_ok=True)
        os.makedirs(os.path.join(self.dir_videos, "videos_agents"), exist_ok=True)
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
        self.dict_idx_channel_to_color_tag_agent: Dict[int, str] = {}
        for name_channel, idx_channel in self.dict_name_channel_to_idx.items():
            if name_channel in self.dict_name_channel_to_color_tag:
                self.dict_idx_channel_to_color_tag[idx_channel] = (
                    self.dict_name_channel_to_color_tag[name_channel]
                )
            else:
                self.dict_idx_channel_to_color_tag[idx_channel] = (
                    self.color_tag_unknown_channel
                )
        for idx_agent_channel, name_channel in enumerate(
            config["list_channels_visual_field"]
        ):
            if name_channel in self.dict_name_channel_to_color_tag:
                self.dict_idx_channel_to_color_tag_agent[idx_agent_channel] = (
                    self.dict_name_channel_to_color_tag[name_channel]
                )
            else:
                self.dict_idx_channel_to_color_tag_agent[idx_agent_channel] = (
                    self.color_tag_unknown_channel
                )
        self.timesteps_render = []
        # Sun Parameters
        self.period_sun: int = config["period_sun"]
        self.method_sun: str = config["method_sun"]
        self.radius_sun_effect: int = config["radius_sun_effect"]
        self.radius_sun_perception: int = config["radius_sun_perception"]
        # Internal dynamics
        self.age_max: int = config["age_max"]
        self.list_death_events: List[str] = config["list_death_events"]
        self.energy_initial: float = config["energy_initial"]
        self.energy_plant: float = config["energy_plant"]
        self.energy_thr_death: float = config["energy_thr_death"]
        self.energy_req_reprod: float = config["energy_req_reprod"]
        self.energy_cost_reprod: float = config["energy_cost_reprod"]
        self.energy_max: float = config["energy_max"]
        self.energy_transfer_loss: float = config.get("energy_transfer_loss", 0.0)
        self.energy_transfer_gain: float = config.get("energy_transfer_gain", 0.0)
        # Other
        self.fill_value: int = self.n_agents_max
        self.novelty_hunger_value_initial = self.age_max
        # Plants Dynamics
        self.proportion_plant_initial: float = config["proportion_plant_initial"]
        self.do_plant_grow_in_fruit_clusters: bool = config[
            "do_plant_grow_in_fruit_clusters"
        ]
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
        # Fruits parameters
        if self.do_fruits:
            self.proportion_fruit_initial: float = config["proportion_fruit_initial"]
            self.p_base_fruit_growth: float = config["p_base_fruit_growth"]
            self.energy_fruit_max_abs: float = config["energy_fruit_max_abs"]
            self.side_cluster_fruits: int = config["side_cluster_fruits"]
            assert (
                self.height % self.side_cluster_fruits == 0
                and self.width % self.side_cluster_fruits == 0
            ), "The side of the cluster of fruits must divide the height and width of the map"
            assert (
                self.side_cluster_fruits % 2 == 1
            ), "The side of the cluster of fruits must be odd"
            self.range_cluster_fruits: int = config["range_cluster_fruits"]
            assert (
                self.range_cluster_fruits <= self.side_cluster_fruits // 2
            ), f"The range of the cluster of fruits must be less than half the side of the cluster, but got {self.range_cluster_fruits} > {self.side_cluster_fruits}//2"
            self.variability_fruits: List[float] = config["variability_fruits"]
            self.mode_variability_fruits: str = config["mode_variability_fruits"]
            self.coords_clusters_to_fruit_id: Dict[
                Tuple[int, int], Tuple[int, float]
            ] = {}
            self.n_clusters_x = self.height // self.side_cluster_fruits
            self.n_clusters_y = self.width // self.side_cluster_fruits
            self.id_ressource_to_map_value: Dict[int, jnp.ndarray] = {
                id_fruit: jnp.zeros(shape=(self.height, self.width))
                for id_fruit in range(4)
            }
            self.id_ressource_to_map_value["plants"] = jnp.ones((self.height, self.width)) * self.energy_plant
            for x in range(self.n_clusters_x):
                for y in range(self.n_clusters_y):
                    # Get the coordinates of the center of the cluster
                    coords_center = (
                        x * self.side_cluster_fruits + self.side_cluster_fruits // 2,
                        y * self.side_cluster_fruits + self.side_cluster_fruits // 2,
                    )
                    # Determine the fruit identifier depending on the cluster
                    if x % 2 == 0 and y % 2 == 0:
                        id_fruit = 0
                    elif x % 2 == 0 and y % 2 == 1:
                        id_fruit = 1
                    elif x % 2 == 1 and y % 2 == 0:
                        id_fruit = 2
                    else:
                        id_fruit = 3
                    # Assign initial value depending on the variability mode
                    w = self.variability_fruits[id_fruit]
                    if self.mode_variability_fruits == "space":
                        value_cluster = (
                            self.energy_fruit_max_abs
                            * (jnp.cos(w * jnp.pi * x))
                            * (jnp.cos(w * jnp.pi * y))
                        )
                    elif self.mode_variability_fruits == "time":
                        value_cluster = (
                            self.energy_fruit_max_abs
                            * (jnp.cos(w * jnp.pi * 0))
                        )
                    else:
                        raise ValueError(f"Unknown mode_variability_fruits: {self.mode_variability_fruits}")
                    self.id_ressource_to_map_value[id_fruit] = (
                        self.id_ressource_to_map_value[id_fruit]
                        .at[
                            coords_center[0]
                            - self.range_cluster_fruits : coords_center[0]
                            + self.range_cluster_fruits
                            + 1,
                            coords_center[1]
                            - self.range_cluster_fruits : coords_center[1]
                            + self.range_cluster_fruits
                            + 1,
                        ]
                        .set(value_cluster)
                    )
                    self.id_ressource_to_map_value["plants"] = (
                        self.id_ressource_to_map_value["plants"]
                        .at[
                            coords_center[0]
                            - self.range_cluster_fruits : coords_center[0]
                            + self.range_cluster_fruits
                            + 1,
                            coords_center[1]
                            - self.range_cluster_fruits : coords_center[1]
                            + self.range_cluster_fruits
                            + 1,
                        ]
                        .add(-self.energy_plant) # Remove the plant energy (no plants grow in fruit clusters)
                    )
                    self.coords_clusters_to_fruit_id[coords_center] = id_fruit
        # ======================== Agent Parameters ========================

        # Observations
        self.list_observations: List[str] = config["list_observations"]
        assert (
            len(self.list_observations) > 0
        ), "The list of observations must be non-empty"

        @dataclass
        class ObservationAgentGridworld(ObservationAgent):
            # The visual field of the agent, of shape (2v+1, 2v+1, n_channels_visual_field)
            if "visual_field" in self.list_observations:
                visual_field: jnp.ndarray  # (2v+1, 2v+1, n_channels_visual_field) in R

            # The energy level of an agent, of shape () and in [0, +inf).
            if "energy" in self.list_observations:
                energy: jnp.ndarray

            # The age of an agent, of shape () and in [0, +inf).
            if "age" in self.list_observations:
                age: jnp.ndarray

            # The novelty hunger of an agent, of shape (4,) and in R+
            if "novelty_hunger" in self.list_observations:
                novelty_hunger: jnp.ndarray

            # The number of childrens of an agent, of shape () and in N
            if "n_childrens" in self.list_observations:
                n_childrens: jnp.ndarray

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
            observation_dict["energy"] = ContinuousSpace(
                shape=(),
                low=0.0,
                high=1.0,
            )
        if "age" in self.list_observations:
            observation_dict["age"] = ContinuousSpace(
                shape=(),
                low=0.0,
                high=1.0,
            )
        if "novelty_hunger" in self.list_observations:
            observation_dict["novelty_hunger"] = ContinuousSpace(
                shape=(4,), low=0.0, high=2.0
            )
        if "n_childrens" in self.list_observations:
            observation_dict["n_childrens"] = ContinuousSpace(
                shape=(), low=0.0, high=None
            )
        self.internal_measures_first_agent: Dict[str, List[jnp.ndarray]] = {
            "energy": [],
            "age": [],
            **{f"novelty_hunger_{id_fruit}": [] for id_fruit in range(4)},
            "n_childrens": [],
        }
        self.observation_space = DictSpace(observation_dict)

        # Actions
        self.list_actions: List[str] = config["list_actions"]
        assert len(self.list_actions) > 0, "The list of actions must be non-empty"
        self.action_to_idx: Dict[str, int] = {
            action: idx for idx, action in enumerate(self.list_actions)
        }
        self.n_actions = len(self.list_actions)

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
        if not self.do_plant_grow_in_fruit_clusters:
            for coords_center, id_fruit in self.coords_clusters_to_fruit_id.items():
                map = map.at[
                    coords_center[0]
                    - self.range_cluster_fruits : coords_center[0]
                    + self.range_cluster_fruits
                    + 1,
                    coords_center[1]
                    - self.range_cluster_fruits : coords_center[1]
                    + self.range_cluster_fruits
                    + 1,
                    idx_plants,
                ].set(0)

        # Initialize the fruits
        if self.do_fruits:
            for coords_center, id_fruit in self.coords_clusters_to_fruit_id.items():
                key_random, subkey = jax.random.split(key_random)
                map = map.at[
                    coords_center[0]
                    - self.range_cluster_fruits : coords_center[0]
                    + self.range_cluster_fruits
                    + 1,
                    coords_center[1]
                    - self.range_cluster_fruits : coords_center[1]
                    + self.range_cluster_fruits
                    + 1,
                    self.dict_name_channel_to_idx[f"fruits_{id_fruit}"],
                ].set(
                    jax.random.bernoulli(
                        key=subkey,
                        p=self.proportion_fruit_initial,
                        shape=(
                            2 * self.range_cluster_fruits + 1,
                            2 * self.range_cluster_fruits + 1,
                        ),
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

        appearance_agents = (
            jnp.zeros((self.n_agents_max, self.config["dim_appearance"]))
            .at[: self.n_agents_initial, :]
            .set(1)
        )

        # Initialize the agents
        agents = AgentGriworld(
            positions_agents=positions_agents,
            orientation_agents=orientation_agents,
            are_existing_agents=are_existing_agents,
            energy_agents=jnp.full((self.n_agents_max,), self.energy_initial),
            age_agents=jnp.zeros(self.n_agents_max),
            appearance_agents=appearance_agents,
            novelty_hunger=jnp.full(
                (self.n_agents_max, 4), self.novelty_hunger_value_initial
            ),
            n_childrens=jnp.zeros(self.n_agents_max, dtype=jnp.int_),
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
        video_agent = jnp.zeros(
            (
                self.n_steps_per_video,
                2 * self.vision_range_agent + 1,
                2 * self.vision_range_agent + 1,
                3,
            )
        )

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
            video_agent=video_agent,
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
            state_species (Optional[StateSpecies]): the state of the species at timestep t, for computing behavior metrics if necessary

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

        # Initialize the measures dictionnary. This will be used to store the measures of the environment at this step.
        state_old = state
        dict_measures_all: Dict[str, jnp.ndarray] = {}
        t = state.timestep

        # ============ (1) Agents interaction with the environment ============
        # Apply the actions of the agents on the environment
        key_random, subkey = jax.random.split(key_random)
        state, dict_measures = self.step_action_agents(
            state=state, actions=actions, key_random=subkey
        )
        dict_measures_all.update(dict_measures)

        # ============ (2) Agents reproduce ============
        key_random, subkey = jax.random.split(key_random)
        state, are_newborns_agents, indexes_parents_agents, dict_measures = (
            self.step_reproduce_agents(state=state, actions=actions, key_random=subkey)
        )
        dict_measures_all.update(dict_measures)

        # ============ (3) Extract the observations of the agents (and some updates) ============

        H, W, C = state.map.shape
        idx_agents = self.dict_name_channel_to_idx["agents"]

        # Recreate the map of agents
        map_agents_new = (
            jnp.zeros((H, W))
            .at[
                state.agents.positions_agents[:, 0],
                state.agents.positions_agents[:, 1],
            ]
            .add(state.agents.are_existing_agents)
        )

        # Recreate the map of appearances
        map_appearances_new = jnp.zeros((H, W, self.config["dim_appearance"]))
        map_appearances_new = map_appearances_new.at[
            state.agents.positions_agents[:, 0],
            state.agents.positions_agents[:, 1],
            :,
        ].add(
            state.agents.appearance_agents * state.agents.are_existing_agents[:, None]
        )
        map_appearances_new /= jnp.maximum(1, map_agents_new[:, :][:, :, None])

        # Update the state
        map_new = state.map.at[:, :, idx_agents].set(map_agents_new)
        map_new = map_new.at[
            :,
            :,
            [
                self.dict_name_channel_to_idx[f"appearance_{i}"]
                for i in range(self.config["dim_appearance"])
            ],
        ].set(map_appearances_new)
        state: StateEnvGridworld = state.replace(
            map=map_new,
            timestep=t + 1,
            agents=state.agents.replace(age_agents=state.agents.age_agents + 1),
        )
        # Extract the observations of the agents
        observations_agents, dict_measures = self.get_observations_agents(state=state)
        video_agent = state.video_agent.at[t % self.n_steps_per_video].set(
            self.get_RGB_map(
                images=observations_agents["visual_field"][0],
                dict_idx_channel_to_color_tag=self.dict_idx_channel_to_color_tag_agent,
            )
        )
        state = state.replace(video_agent=video_agent)
        dict_measures_all.update(dict_measures)

        # ============ (4) Get the ecological information ============
        are_just_dead_agents = state.agents.are_existing_agents & (
            ~state.agents.are_existing_agents
            | (state.agents.age_agents < state.agents.age_agents)
        )
        eco_information = EcoInformation(
            are_newborns_agents=are_newborns_agents,
            indexes_parents=indexes_parents_agents,
            are_just_dead_agents=are_just_dead_agents,
        )

        # ============ (5) Check if the environment is done ============
        if self.is_terminal:
            done = ~jnp.any(state.agents.are_existing_agents)
        else:
            done = False

        # ============ (6) Compute the metrics ============

        # Compute some measures
        dict_measures = self.compute_measures(
            state=state_old,
            actions=actions,
            state_new=state,
            key_random=subkey,
            state_species=state_species,
        )
        dict_measures_all.update(dict_measures)

        # Update and compute the metrics
        state, dict_metrics = self.compute_metrics(
            state=state_old, state_new=state, dict_measures=dict_measures_all
        )
        info = {"metrics": dict_metrics}

        # ============ (7) Manage the video ============
        # Reset the video to empty if t = 0 mod n_steps_per_video
        video = jax.lax.cond(
            t % self.n_steps_per_video == 0,
            lambda _: jnp.zeros((self.n_steps_per_video, H, W, 3)),
            lambda _: state.video,
            operand=None,
        )
        video_agent = jax.lax.cond(
            t % self.n_steps_per_video == 0,
            lambda _: jnp.zeros(
                (
                    self.n_steps_per_video,
                    2 * self.vision_range_agent + 1,
                    2 * self.vision_range_agent + 1,
                    3,
                )
            ),
            lambda _: state.video_agent,
            operand=None,
        )
        # Add the new frame to the video
        video = state.video.at[t % self.n_steps_per_video].set(
            self.get_RGB_map(
                images=state.map,
                dict_idx_channel_to_color_tag=self.dict_idx_channel_to_color_tag,
            )
        )
        state = state.replace(video=video)

        # Return the new state and observations
        return (
            state,
            observations_agents,
            eco_information,
            done,
            info,
        )

    def get_observation_space(self) -> DictSpace:
        return self.observation_space

    def get_action_space(self) -> ContinuousSpace:
        return DiscreteSpace(n=self.n_actions)

    def render(self, state: StateEnvGridworld, force_render: bool = False) -> None:
        """The rendering function of the environment. It saves the RGB map of the environment as a video."""
        t = state.timestep
        self.timesteps_render.append(t)

        # Save videos of simulation and agent0 visual field
        def save_video(video: jnp.ndarray, filename: str) -> None:
            tqdm.write(f"Rendering video : {filename}...")
            video_writer = VideoRecorder(filename=filename, fps=self.fps_video)
            for t_ in range(t, t + self.n_steps_per_video):
                t_ = t_ % self.n_steps_per_video
                if t_ >= t:
                    continue
                image = video[t_]
                image = self.upscale_image(image)
                video_writer.add(image)
            video_writer.close()

        if (
            t >= self.n_steps_per_video
            and t >= self.t_last_video_rendered + self.n_steps_min_between_videos
        ) or force_render:
            self.t_last_video_rendered = t
            # Render map
            if self.do_video:
                save_video(
                    filename=f"{self.dir_videos}/video_{max(0, t-self.n_steps_per_video)}_to_{t}.mp4",
                    video=state.video,
                )
            if self.do_agent_video:
                save_video(
                    filename=f"{self.dir_videos}/videos_agents/video_agent_{max(0, t-self.n_steps_per_video)}_to_{t}.mp4",
                    video=state.video_agent,
                )

        # Log the life of the first agent
        output_dir = "./logs/curves_agent/"
        os.makedirs(output_dir, exist_ok=True)
        # Add the measures of the first agent to the internal measures
        if state.agents.are_existing_agents[0]:
            self.internal_measures_first_agent["energy"].append(
                state.agents.energy_agents[0]
            )
            self.internal_measures_first_agent["age"].append(state.agents.age_agents[0])
            for id_fruit in range(4):
                self.internal_measures_first_agent[f"novelty_hunger_{id_fruit}"].append(
                    state.agents.novelty_hunger[0, id_fruit]
                )
            self.internal_measures_first_agent["n_childrens"].append(
                state.agents.n_childrens[0]
            )
        else:
            for measure_name in self.internal_measures_first_agent.keys():
                self.internal_measures_first_agent[measure_name].append(jnp.nan)
        # Loop through the dictionary and create a plot for each measure
        for measure_name, values in self.internal_measures_first_agent.items():
            values = jnp.array(values)
            plt.figure()  # Create a new figure
            plt.plot(
                self.timesteps_render, values, label=measure_name, marker="o"
            )  # Plot the values
            plt.title(f"Curve for {measure_name}")  # Set the title
            plt.xlabel("Timestep")  # Label x-axis
            plt.ylabel("Value")  # Label y-axis
            plt.legend()  # Show legend
            plt.grid(True)  # Show grid
            file_path = os.path.join(
                output_dir, f"curve_{measure_name}.png"
            )  # Define file path
            plt.savefig(file_path)  # Save the plot as a PNG file
            plt.close()  # Close the plot to free memory

        # Signal if env terminated
        if ~jnp.any(state.agents.are_existing_agents):
            tqdm.write("Environment is done.")

    # ================== Helper functions ==================

    def get_RGB_map(
        self, images: jnp.ndarray, dict_idx_channel_to_color_tag: Dict[int, str]
    ) -> jnp.ndarray:
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
        ), f"Unknown color tag: {self.color_tag_background}. Pick among {list(DICT_COLOR_TAG_TO_RGB.keys())}"
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
        for channel_idx, color_tag in dict_idx_channel_to_color_tag.items():
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
        logits_plants = (
            self.logit_p_base_plant_growth * (1 - map_plants)
            + (1 - self.logit_p_base_plant_death) * map_plants
            + self.factor_sun_effect * map_sun
            + self.factor_plant_reproduction * map_n_plant_in_radius_plant_reproduction
            - self.factor_plant_asphyxia * map_n_plant_in_radius_plant_asphyxia
        )
        logits_plants = jnp.clip(logits_plants, -10, 10)
        map_plants_probs = sigmoid(x=logits_plants)
        key_random, subkey = jax.random.split(key_random)
        map_plants = jax.random.bernoulli(
            key=subkey,
            p=map_plants_probs,
            shape=map_plants.shape,
        )
        if not self.do_plant_grow_in_fruit_clusters:
            for coords_center, id_fruit in self.coords_clusters_to_fruit_id.items():
                map_plants = map_plants.at[
                    coords_center[0]
                    - self.range_cluster_fruits : coords_center[0]
                    + self.range_cluster_fruits
                    + 1,
                    coords_center[1]
                    - self.range_cluster_fruits : coords_center[1]
                    + self.range_cluster_fruits
                    + 1,
                ].set(0)
        return state.replace(map=state.map.at[:, :, idx_plants].set(map_plants))

    def step_grow_fruits(
        self, state: StateEnvGridworld, key_random: jnp.ndarray
    ) -> StateEnvGridworld:
        """Grow fruits."""
        map = state.map
        for coords_center, id_fruit in self.coords_clusters_to_fruit_id.items():
            key_random, subkey = jax.random.split(key_random)
            idx_fruit_i = self.dict_name_channel_to_idx[f"fruits_{id_fruit}"]
            map = map.at[
                coords_center[0]
                - self.range_cluster_fruits : coords_center[0]
                + self.range_cluster_fruits
                + 1,
                coords_center[1]
                - self.range_cluster_fruits : coords_center[1]
                + self.range_cluster_fruits
                + 1,
                idx_fruit_i,
            ].add(
                jax.random.bernoulli(
                    key=subkey,
                    p=self.p_base_fruit_growth,
                    shape=(
                        2 * self.range_cluster_fruits + 1,
                        2 * self.range_cluster_fruits + 1,
                    ),
                )
            )
            map = map.at[:, :, idx_fruit_i].set(jnp.clip(map[:, :, idx_fruit_i], 0, 1))
        return state.replace(map=map)

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

    def get_map_ressource_energy(
        self, state: StateEnvGridworld, id_ressource: int, key_random: jnp.ndarray
    ) -> jnp.ndarray:
        """Get the map of the energy of the fruits, as an array of shape (H, W).

        Args:
            state (StateEnvGridworld): the state of the environment
            id_fruit (int): the indicator of the ressource, between 0 and 3 for fruits, or "plants" for plants
            key_random (jnp.ndarray): the random key

        Returns:
            jnp.ndarray: the map of the energy of the fruits, as an array of shape (H, W)
        """
        if id_ressource == "plants":
            idx_plants = self.dict_name_channel_to_idx["plants"]
            map_plants = state.map[:, :, idx_plants]
            return map_plants * self.energy_plant
        idx_fruit_i = self.dict_name_channel_to_idx[f"fruits_{id_ressource}"]
        map_fruits = state.map[:, :, idx_fruit_i]  # (H, W), whether there is a fruit
        if self.mode_variability_fruits == "space":
            map_value_fruit_i = self.id_ressource_to_map_value[id_ressource]
            return map_fruits * map_value_fruit_i
        elif self.mode_variability_fruits == "time":
            w = self.variability_fruits[id_ressource]
            t = state.timestep
            value_fruit = self.energy_fruit_max_abs * jnp.cos(
                2 * jnp.pi * w * t / self.age_max
            )
            return map_fruits * value_fruit
        else:
            raise ValueError(
                f"Unknown mode_variability_fruits: {self.mode_variability_fruits}"
            )

    def get_single_agent_new_position_and_orientation(
        self,
        agent_position: jnp.ndarray,
        agent_orientation: jnp.ndarray,
        action: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get the new position and orientation of a single agent.
        Args:
            agent_position (jnp.ndarray): the position of the agent, of shape (2,)
            agent_orientation (jnp.ndarray): the orientation of the agent, of shape ()
            action (int): the action of the agent, an integer between 0 and self.n_actions - 1

        Returns:
            jnp.ndarray: the new position of the agent, of shape (2,)
            jnp.ndarray: the new orientation of the agent, of shape ()
        """
        H, W = self.height, self.width
        condlist: List[jnp.ndarray] = []
        choicelist_position: List[jnp.ndarray] = []
        choicelist_orientation: List[jnp.ndarray] = []
        for name_action_move in ["forward", "backward", "left", "right"]:
            # Check if the action is a move action
            if name_action_move in self.list_actions:
                # Add the action to the list of possible actions related to moving
                condlist.append(action == self.action_to_idx[name_action_move])
                # Add the new orientation to the list of possible orientations
                if name_action_move == "forward":
                    direction = 0
                elif name_action_move == "backward":
                    direction = 2
                elif name_action_move == "left":
                    direction = 1
                elif name_action_move == "right":
                    direction = 3
                else:
                    raise ValueError(f"Incorrect implementation")
                agent_orientation_new = (agent_orientation + direction) % 4
                choicelist_orientation.append(agent_orientation_new)
                # Add the new position to the list of possible positions
                angle_new = agent_orientation_new * jnp.pi / 2
                d_position = jnp.array(
                    [-jnp.cos(angle_new), -jnp.sin(angle_new)]
                ).astype(jnp.int32)
                agent_position_new = agent_position + d_position
                agent_position_new = agent_position_new % jnp.array([H, W])
                choicelist_position.append(agent_position_new)

        # Select the new position and orientation of the agent based on the action
        agent_position_new = jnp.select(
            condlist=condlist,
            choicelist=choicelist_position,
            default=agent_position,
        )
        agent_orientation_new = jnp.select(
            condlist=condlist,
            choicelist=choicelist_orientation,
            default=agent_orientation,
        )
        return agent_position_new, agent_orientation_new

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

        # ====== Get which agents are eating ======
        if "eat" in self.list_actions:
            # If "eat" is in the list of actions, then the agents have to take the action "eat" to eat
            are_agents_eating = state.agents.are_existing_agents & (
                actions == self.action_to_idx["eat"]
            )
        else:
            # Else, agents are eating passively
            raise ValueError(
                "The action 'eat' must be in the list of actions in this code state"
            )
            are_agents_eating = state.agents.are_existing_agents

        positions_agents = state.agents.positions_agents
        orientation_agents = state.agents.orientation_agents
        map_n_agents_try_eating = (
            jnp.zeros_like(map_agents)
            .at[
                positions_agents[:, 0], positions_agents[:, 1]
            ]  # we use the position at s_{t-1} since anyway moving agents will not eat
            .add(are_agents_eating)
        )  # map of the number of (existing) agents trying to eat at each cell

        # ====== Compute the energy bonus obtainable on each tile ======
        # Add the energy bonus from the plants
        map_food_energy_bonus_available_per_agent = self.energy_plant * map_plants
        # Add the energy bonus from the fruits
        if self.do_fruits:
            for id_fruit in range(4):
                map_fruit_energy = self.get_map_ressource_energy(
                    state=state, id_ressource=id_fruit, key_random=key_random
                )
                map_food_energy_bonus_available_per_agent += map_fruit_energy
        # Compute the energy bonus obtainable by each agent
        map_food_energy_bonus_available_per_agent /= jnp.maximum(
            1, map_n_agents_try_eating
        )  # divide by the number of agents trying to eat at each cell
        food_energy_bonus_available_per_agent = (
            map_food_energy_bonus_available_per_agent[
                positions_agents[:, 0], positions_agents[:, 1]
            ]
        )
        food_energy_bonus = food_energy_bonus_available_per_agent * are_agents_eating
        energy_agents_new = state.agents.energy_agents + food_energy_bonus
        
        # ===== Update novelty hunger (and compute some eating measures) =====
        novelty_hunger_new = state.agents.novelty_hunger  # (n, 4)
        if self.do_fruits:
            for id_fruit in range(4):
                idx_fruit = self.dict_name_channel_to_idx[f"fruits_{id_fruit}"]
                map_fruit = state.map[..., idx_fruit]
                are_agents_eating_fruit_i = are_agents_eating & (
                    map_fruit[positions_agents[:, 0], positions_agents[:, 1]] == 1.0
                )
                if f"eating" in self.names_measures:
                    dict_measures.update(self.compute_do_eat_measures(
                        are_agent_eating_ressource=are_agents_eating_fruit_i,
                        food_energy_bonus_available_per_agent=food_energy_bonus_available_per_agent,
                        ressource_name=f"fruits_{id_fruit}",
                    ))
                # Update the novelty hunger of fruit i
                novelty_hunger_new = novelty_hunger_new.at[:, id_fruit].set(
                    jnp.where(
                        are_agents_eating_fruit_i,
                        0,  # if the agent eats in a tile containing fruit i, then the novelty hunger of fruit i is reset
                        novelty_hunger_new[:, id_fruit]
                        + 1,  # else, the novelty hunger of fruit i is incremented
                    )
                )
                
        if "eating" in self.names_measures:
            are_agent_eating_plants = are_agents_eating & (
                map_plants[positions_agents[:, 0], positions_agents[:, 1]] == 1.0
            )
            dict_measures.update(self.compute_do_eat_measures(
                are_agent_eating_ressource=are_agent_eating_plants,
                food_energy_bonus_available_per_agent=food_energy_bonus_available_per_agent,
                ressource_name="plants",
            ))
            dict_measures["minimap_eat"] = jnp.zeros((self.height, self.width)).at[
                positions_agents[:, 0], positions_agents[:, 1]
            ].set(-1 + 2*are_agents_eating) # 0 if no agents, -1 if agents not eating, 1 if agents eating
            
        if "amount_food_eaten" in self.names_measures:
            dict_measures["amount_food_eaten"] = food_energy_bonus

        # ====== Remove plants and fruits that have been eaten ======
        map = state.map
        indexes_plants_and_fruits = [idx_plants] + [
            self.dict_name_channel_to_idx[f"fruits_{i}"]
            for i in range(4)
            if self.do_fruits
        ]
        for idx in indexes_plants_and_fruits:
            map = map.at[:, :, idx].set(
                jnp.clip(
                    map[:, :, idx] - map_n_agents_try_eating * map[:, :, idx], 0, 1
                )
            )
        state = state.replace(map=map)

        # ====== Compute the new positions and orientations of all the agents ======
        get_many_agents_new_position_and_orientation = jax.vmap(
            self.get_single_agent_new_position_and_orientation, in_axes=(0, 0, 0)
        )
        positions_agents_new, orientation_agents_new = (
            get_many_agents_new_position_and_orientation(
                positions_agents,
                orientation_agents,
                actions,
            )
        )

        # ====== Handle any energy transfer actions ======
        if "transfer" in self.list_actions:
            # Check if any agents are transferring energy
            are_agents_transferring = state.agents.are_existing_agents & (
                actions == self.action_to_idx["transfer"]
            )
            # compute pairwise manhattan distance between agents
            distances = jnp.sum(
                jnp.abs(positions_agents_new[:, None] - positions_agents_new[None, :]),
                axis=-1,
            )

            def per_agent_helper_fn(i):
                # don't allow both eating and transferring in one timestep
                is_transfer = (
                    state.agents.are_existing_agents[i] & are_agents_transferring[i]
                )

                # energy can only be transferred to agents on the same or a directly adjacent tile
                are_receiving = state.agents.are_existing_agents & (
                    distances[i, :] <= 1
                )

                # ensure there are other agents to transfer to, i.e. don't allow self-transfer
                is_transfer &= jnp.sum(are_receiving) > 1

                loss = is_transfer * self.energy_transfer_loss
                gain = (
                    is_transfer
                    * self.energy_transfer_gain
                    / jnp.maximum(1, jnp.sum(are_receiving))
                )
                return jnp.zeros_like(state.agents.energy_agents).at[i].add(-loss) + (
                    gain * are_receiving
                ).astype(jnp.float32)

            transfer_delta_energy = jax.vmap(per_agent_helper_fn, in_axes=0)(
                jnp.arange(state.agents.energy_agents.size)
            ).sum(axis=0)
            energy_agents_new += transfer_delta_energy

            # log the net energy transfer (should always be >= 0)
            if "net_energy_transfer_per_capita" in self.names_measures:
                dict_measures["net_energy_transfer_per_capita"] = (
                    transfer_delta_energy.sum() / state.agents.are_existing_agents.sum()
                )

        # ====== Update the physical status of the agents ======
        energy_agents_new -= 1
        are_existing_agents_new = state.agents.are_existing_agents
        if "age" in self.list_death_events:
            are_existing_agents_new &= state.agents.age_agents < self.age_max
        if "energy" in self.list_death_events:
            are_existing_agents_new &= energy_agents_new > self.energy_thr_death
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
            novelty_hunger=novelty_hunger_new,
        )
        state = state.replace(
            agents=agents_new,
        )

        # Update the sun
        key_random, subkey = jax.random.split(key_random)
        state = self.step_update_sun(state=state, key_random=subkey)
        # Grow plants
        key_random, subkey = jax.random.split(key_random)
        state = self.step_grow_plants(state=state, key_random=subkey)
        # Grow fruits
        if self.do_fruits:
            key_random, subkey = jax.random.split(key_random)
            state = self.step_grow_fruits(state=state, key_random=subkey)
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
            state.agents.energy_agents > self.energy_req_reprod
        ) & are_existing_agents
        if "reproduce" in self.list_actions:
            are_agents_trying_reprod &= actions == self.action_to_idx["reproduce"]

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
        positions_agents_new = state.agents.positions_agents.at[
            indices_newborn_agents_FILLED
        ].set(state.agents.positions_agents[indices_had_reproduced_FILLED])
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
        novelty_hunger_new = state.agents.novelty_hunger.at[
            indices_newborn_agents_FILLED
        ].set(self.novelty_hunger_value_initial)
        n_childrens_new = state.agents.n_childrens.at[
            indices_had_reproduced_FILLED
        ].set(0)
        n_childrens_new += are_agents_reproducing

        # Update the state
        agents_new = state.agents.replace(
            energy_agents=energy_agents_new,
            are_existing_agents=are_existing_agents_new,
            age_agents=age_agents_new,
            positions_agents=positions_agents_new,
            appearance_agents=appearance_agents_new,
            novelty_hunger=novelty_hunger_new,
            n_childrens=n_childrens_new,
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
            map_vis_field = state.map[:, :, self.list_indexes_channels_visual_field]

            # Get the visual field of the agent
            visual_field_x = agents.positions_agents[0] + self.grid_indexes_vision_x
            visual_field_y = agents.positions_agents[1] + self.grid_indexes_vision_y
            vis_field = map_vis_field[
                visual_field_x % H,
                visual_field_y % W,
            ]  # (2 * self.vision_radius + 1, 2 * self.vision_radius + 1, C_map)

            # print(f"{map_vis_field[..., 0]=}")
            # print(f"{agents.positions_agents=}")
            # print(f"{self.grid_indexes_vision_x=}")
            # print(f"{self.grid_indexes_vision_y=}")
            # print(f"{visual_field_x=}")
            # print(f"{visual_field_y=}")
            # print(f"{vis_field[..., 0]=}")

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
                    jnp.rot90(
                        vis_field, k=3, axes=(0, 1)
                    ),  # if we turn left, the observation should be rotated right, eg 270 degrees
                    jnp.rot90(vis_field, k=2, axes=(0, 1)),
                    jnp.rot90(vis_field, k=1, axes=(0, 1)),
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
        if "novelty_hunger" in self.list_observations:
            dict_observations["novelty_hunger"] = (
                state.agents.novelty_hunger / self.age_max
            )
        if "n_childrens" in self.list_observations:
            dict_observations["n_childrens"] = state.agents.n_childrens
        if "visual_field" in self.list_observations:
            dict_observations["visual_field"] = jax.vmap(get_single_agent_visual_field)(
                state.agents
            )

        # Compute some observations-related measures
        dict_measures = {}  # None for now

        # print(f"Map : {state.map[..., 0]}")
        # print(f"Agents positions : {state.positions_agents}")
        # print(f"Agents orientations : {state.orientation_agents}")
        # print(f"First agent obs : {observations.visual_field[0, ..., 2]}, shape : {observations.visual_field[0, ...].shape}")
        # raise ValueError("Stop")

        return dict_observations, dict_measures

    # ================== Measures and metrics ==================

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
            elif name_measure == "values_fruits" and self.do_fruits:
                if self.mode_variability_fruits == "time":
                    for id_fruit in range(4):
                        w = self.variability_fruits[id_fruit]
                        t = state.timestep
                        value_fruit = self.energy_fruit_max_abs * jnp.cos(
                            2 * jnp.pi * w * t / self.age_max
                        )
                        dict_measures[f"value_fruits {id_fruit}/value_fruits"] = (
                            value_fruit
                        )
                elif self.mode_variability_fruits == "space":
                    minimap_value_ressource = jnp.zeros((self.height, self.width))  # set the value as energy_plant (plant values in cluster will be removed below)
                    for key_ressource, map_value in self.id_ressource_to_map_value.items():
                        minimap_value_ressource += map_value
                    dict_measures[f"minimap_value_ressource"] = minimap_value_ressource
                        
                else:
                    raise ValueError(
                        f"Unknown mode_variability_fruits: {self.mode_variability_fruits}"
                    )
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
            elif name_measure == "novelty_hunger" and self.do_fruits:
                for id_fruit in range(4):
                    dict_measures[f"novelty_hunger_{id_fruit}"] = (
                        state.agents.novelty_hunger[:, id_fruit]
                    )
            elif name_measure == "n_childrens":
                dict_measures[name_measure] = state.agents.n_childrens
            elif name_measure == "x":
                dict_measures[name_measure] = state.agents.positions_agents[:, 0]
            elif name_measure == "y":
                dict_measures[name_measure] = state.agents.positions_agents[:, 1]
            elif name_measure == "appearance":
                for idx_appearance in range(self.config["dim_appearance"]):
                    dict_measures[f"appearance_{idx_appearance}"] = (
                        state.agents.appearance_agents[:, idx_appearance]
                    )
            # Behavior measures (requires state_species)
            elif name_measure in self.config["metrics"]["measures"]["behavior"]:
                assert isinstance(
                    self.agent_species, AgentSpecies
                ), f"For behavior measure, you need to give an agent species as attribute of the env after both creation : env.agent_species = agent_species"
                dict_measures.update(
                    self.compute_behavior_measure(
                        state_species=state_species,
                        key_random=key_random,
                        name_measure=name_measure,
                    )
                )
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
                # Skip the measures that are not of shape (n_max_agents, ...)
                if measures.shape[0] != self.n_agents_max:
                    continue
                # Expand the mask to the shape of the measures
                mask = state.agents.are_existing_agents
                for _ in range(measures.ndim - 1):
                    mask = jnp.expand_dims(mask, axis=-1)
                # Turn to NaN the measures of the agents that are not existing
                dict_measures[name_measure] = jnp.where(
                    mask,
                    measures,
                    jnp.nan,
                )

        # Aggregate the measures over the lifespan
        are_just_dead_agents = state_new.agents.are_existing_agents & (
            ~state.agents.are_existing_agents
            | (state_new.agents.age_agents < state.agents.age_agents)
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
        dict_measures_and_metrics_lifespan = {**dict_measures, **dict_metrics_lifespan}
        for agg, metrics in zip(self.aggregators_population, state.metrics_population):
            new_metrics = agg.update_metrics(
                metrics=metrics,
                dict_measures=dict_measures_and_metrics_lifespan,
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

    def compute_behavior_measure(
        self,
        state_species: StateSpecies,
        key_random: jnp.ndarray,
        name_measure: str,
    ) -> Dict[str, jnp.ndarray]:
        """Compute a behavior measure.

        Args:
            state_species (StateSpecies): the state of the species
            react_fn (Callable[ [ StateSpecies, ObservationAgent, EcoInformation, jnp.ndarray, ], Tuple[ StateSpecies, ActionAgent, Dict[str, jnp.ndarray], ], ]): the function that computes the reaction of the species to the environment
            key_random (jnp.ndarray): the random key
            name_measure (str): the name of the measure to compute

        Returns:
            Dict[str, jnp.ndarray]: a dictionary of the behavior measures
        """
        measures: Dict[str, jnp.ndarray] = {}

        if name_measure == "appetite":
            if "plants" in self.dict_name_channel_to_idx_visual_field:
                idx_plant = self.dict_name_channel_to_idx_visual_field["plants"]
                n = self.n_agents_max
                v = self.vision_range_agent
                names_action_to_xy = {
                    "forward": (jnp.full(n, v - 1), jnp.full(n, v)),
                    "left": (jnp.full(n, v), jnp.full(n, v - 1)),
                    "right": (jnp.full(n, v), jnp.full(n, v + 1)),
                    "backward": (jnp.full(n, v + 1), jnp.full(n, v)),
                }
                names_action_to_xy = {
                    key: xy
                    for key, xy in names_action_to_xy.items()
                    if key in self.list_actions
                }
                eco_information = EcoInformation(
                    are_newborns_agents=jnp.full(n, False),
                    indexes_parents=jnp.full((n, 1), self.fill_value),
                    are_just_dead_agents=jnp.full(n, False),
                )

                appetites = jnp.full(n, 0.0, dtype=jnp.float32)
                for name_action, (x, y) in names_action_to_xy.items():
                    actions_appetite = jnp.full(n, self.action_to_idx[name_action])
                    visual_field = (
                        jnp.zeros(
                            (
                                n,
                                2 * v + 1,
                                2 * v + 1,
                                len(self.list_indexes_channels_visual_field),
                            )
                        )
                        .at[
                            jnp.arange(n),
                            x,
                            y,
                            idx_plant,
                        ]
                        .set(1)
                    )
                    assert (
                        "visual_field" in self.list_observations
                    ), "Visual field is required to compute appetite"
                    obs = {"visual_field": visual_field}
                    if "age" in self.list_observations:
                        obs["age"] = jnp.full(n, 50) / self.age_max
                    if "energy" in self.list_observations:
                        obs["energy"] = (
                            jnp.full(n, 5) / self.energy_max
                        )  # very low energy
                    if "n_childrens" in self.list_observations:
                        obs["n_childrens"] = jnp.zeros(n)
                    if "novelty_hunger" in self.list_observations:
                        obs["novelty_hunger"] = jnp.zeros((n, 4))
                    key_random, subkey = jax.random.split(key_random)
                    _, actions, _ = self.agent_species.react(
                        state_species,
                        obs,
                        eco_information,
                        subkey,
                    )
                    appetites += (actions == actions_appetite).astype(jnp.float32)
                appetites = appetites / len(names_action_to_xy)
                measures["appetite"] = appetites

        else:
            raise ValueError(f"Unknown behavior measure: {name_measure}")

        return measures

    def compute_on_render_behavior_measures(
        self,
        state_species: StateSpecies,
        key_random: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray]:
        dict_measures = {}
        for name_measure in self.behavior_measures_on_render:
            assert (
                name_measure not in self.names_measures
            ), f"Behavior measure {name_measure} is both measured at render and step time, this should be avoided."
            dict_measures.update(
                self.compute_behavior_measure(
                    state_species=state_species,
                    key_random=key_random,
                    name_measure=name_measure,
                )
            )
        return dict_measures

    def compute_do_eat_measures(
        self,
        are_agent_eating_ressource : jnp.ndarray,
        food_energy_bonus_available_per_agent : jnp.ndarray,
        ressource_name: str
        ) -> Dict[str, jnp.ndarray]:
        """Compute, for each ressource, the measures related to eating behavior.
        It computes the following :
            eating {ressource name}/eating : P(agent eat ressource)
            eating {ressource name} if positive/eating : P(agent eat ressource | ressource is positive)
            eating no {ressource name} if negative/eating : P(agent dont eat ressource | ressource is negative)
            eating {ressource name} iff ressource/eating : P(agent eat ressource iff ressource is positive)
            
        Args:
            are_agent_eating_ressource (jnp.ndarray): whether the agent is eating the ressource
            food_energy_bonus_available_per_agent (jnp.ndarray): the energy bonus obtainable by each agent
            ressource_name (str): the name of the ressource
            
        Returns:
            Dict[str, jnp.ndarray]: the measures related to eating behavior
        """
        measures = {f"eating {ressource_name}/eating": are_agent_eating_ressource}
        measures[f"eating {ressource_name} if positive/eating"] = jnp.select(
            condlist=[
                food_energy_bonus_available_per_agent > 0,
                food_energy_bonus_available_per_agent <= 0,
            ],
            choicelist=[are_agent_eating_ressource, jnp.nan],
        )
        measures[f"eating no {ressource_name} if negative/eating"] = jnp.select(
            condlist=[
                food_energy_bonus_available_per_agent > 0,
                food_energy_bonus_available_per_agent <= 0,
            ],
            choicelist=[jnp.nan, ~are_agent_eating_ressource],
        )
        measures[f"eating {ressource_name} iff ressource/eating"] = are_agent_eating_ressource * (
            food_energy_bonus_available_per_agent > 0
        ) + (~are_agent_eating_ressource) * (food_energy_bonus_available_per_agent <= 0)
        return measures
        
        
    def obs_idx_to_meaning(self) -> Dict[str, str]:
        # Assume MLP model with visual field in first position
        idx_plant = self.dict_name_channel_to_idx["plants"]
        v = self.vision_range_agent
        side = 2 * v + 1
        res = {}
        for idx in range(len(self.list_indexes_channels_visual_field)):
            for x in range(side):
                for y in range(side):
                    res[side**2 * idx + x * side + y] = f"VisualField_{idx}_({x},{y})"
        return res
        return {
            idx_plant * side**2 + (v - 1) * side + v: "PlantOnForward",
            idx_plant * side**2 + v * side + v - 1: "PlantOnLeft",
            idx_plant * side**2 + v * side + v + 1: "PlantOnRight",
            idx_plant * side**2 + (v + 1) * side + v: "PlantOnBackward",
            idx_plant * side**2 + v * side + v: "PlantOnCenter",
        }

    def action_idx_to_meaning(self) -> Dict[int, str]:
        return {idx: action_str for action_str, idx in self.action_to_idx.items()}


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
