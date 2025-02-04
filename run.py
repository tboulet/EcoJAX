# Logging
import os
import cProfile
from ecojax.core.eco_loop import eco_loop
from ecojax.loggers import BaseLogger
from ecojax.loggers.cli import LoggerCLI
from ecojax.loggers.csv import LoggerCSV
from ecojax.loggers.snakeviz import LoggerSnakeviz
from ecojax.loggers.tensorboard import LoggerTensorboard
from ecojax.loggers.wandb import LoggerWandB


# Config system
import hydra
from omegaconf import OmegaConf, DictConfig
from ecojax.metrics.utils import get_dict_metrics_by_type
from ecojax.register_hydra import register_hydra_resolvers
from ecojax.types import ObservationAgent, StateEnv, StateGlobal, StateSpecies

register_hydra_resolvers()

# Utils
from tqdm import tqdm
import datetime
from time import time, sleep
from typing import Any, Dict, List, Type

# ML libraries
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from flax import struct

# Project imports
from ecojax.environment import env_name_to_EnvClass
from ecojax.agents import agent_name_to_AgentSpeciesClass
from ecojax.models import model_name_to_ModelClass
from ecojax.core.eco_info import EcoInformation
from ecojax.video import VideoRecorder
from ecojax.time_measure import RuntimeMeter
from ecojax.utils import check_jax_device, is_array, is_scalar, try_get_seed


@hydra.main(config_path="configs", config_name="default.yaml")
def main(config: DictConfig):

    # Print informations
    print(f"Current working directory: {os.getcwd()}")
    check_jax_device()
    print("Configuration used :")
    print(OmegaConf.to_yaml(config))
    config = OmegaConf.to_container(config, resolve=True)

    # Run in a snakeviz profile
    runner = Runner(config)
    runner.run()

    # ================ Configuration ================


class Runner:
    def __init__(self, config: Dict):
        self.config = config

    def run(self):

        # Main run's components
        env_name = self.config["env"]["name"]
        agent_species_name = self.config["agents"]["name"]
        model_name = self.config["model"]["name"]

        # ================ Initialization ================

        # Seed
        seed = try_get_seed(self.config)
        print(f"Using seed: {seed}")
        np.random.seed(seed)
        key_random = random.PRNGKey(seed)

        # Run name
        run_name = f"[{agent_species_name}_{model_name}_{env_name}]_{datetime.datetime.now().strftime('%dth%mmo_%Hh%Mmin%Ss')}_seed{seed}"
        run_name = self.config.get("run_name", run_name)
        if hasattr(self.config, "benchmark_name"):
            print(f"Running in benchmark {self.config['benchmark_name']}")
        self.config["run_name"] = run_name

        # Create the env
        EnvClass = env_name_to_EnvClass[env_name]
        if not self.config["do_global_log"]:
            dir_videos = f"./logs/{run_name}/videos"
        else:
            dir_videos = "./logs/videos"
        self.config["env"]["metrics"]["config_video"][
            "dir_videos"
        ] = dir_videos  # I add this line to force the dir_videos to be the one I want
        env = EnvClass(
            config=self.config["env"],
            n_agents_max=self.config["n_agents_max"],
            n_agents_initial=self.config["n_agents_initial"],
        )
        observation_space = env.get_observation_space()
        action_space = env.get_action_space()

        # Create the model
        ModelClass = model_name_to_ModelClass[model_name]

        # Create the agent's species
        AgentSpeciesClass = agent_name_to_AgentSpeciesClass[agent_species_name]
        agent_species = AgentSpeciesClass(
            config=self.config["agents"],
            n_agents_max=self.config["n_agents_max"],
            n_agents_initial=self.config["n_agents_initial"],
            observation_space=observation_space,
            action_space=action_space,
            model_class=ModelClass,
            config_model=self.config["model"],
        )
        env.agent_species = agent_species # give the react function to the environment (for behavior measures)
        agent_species.env = env # give the environment to the agent_species
        
        # ============== Simulation loop ===============
        key_random, subkey = random.split(key_random)
        eco_loop(
            env=env,
            agent_species=agent_species,
            config=self.config,
            key_random=subkey,
        )


if __name__ == "__main__":
    main()
