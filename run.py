# Logging
import os
import cProfile
from ecojax.loggers import BaseLogger
from ecojax.loggers.cli import LoggerCLI
from ecojax.loggers.csv import LoggerCSV
from ecojax.loggers.snakeviz import LoggerSnakeviz
from ecojax.loggers.tensorboard import LoggerTensorboard
from ecojax.loggers.wandb import LoggerWandB


# Config system
import hydra
from omegaconf import OmegaConf, DictConfig
from ecojax.metrics.utils import get_dicts_metrics
from ecojax.register_hydra import register_hydra_resolvers

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

# Project imports
from ecojax.environment import env_name_to_EnvClass
from ecojax.agents import agent_name_to_AgentSpeciesClass
from ecojax.models import model_name_to_ModelClass
from ecojax.video import VideoRecorder
from ecojax.time_measure import RuntimeMeter
from ecojax.utils import check_jax_device, is_array, is_scalar, try_get_seed

check_jax_device()


@hydra.main(config_path="configs", config_name="default.yaml")
def main(config: DictConfig):
    # Load the configuration
    print("Configuration used :")
    print(OmegaConf.to_yaml(config))
    config = OmegaConf.to_container(config, resolve=True)
    # Run in a snakeviz profile
    run = Runner(config)
    do_snakeviz: bool = config["do_snakeviz"]

    if not do_snakeviz:
        run.run()
    else:
        with cProfile.Profile() as pr:
            run.run()
        pr.dump_stats("logs/profile_stats.prof")
        print("Profile stats dumped to logs/profile_stats.prof")

    # ================ Configuration ================


class Runner:
    def __init__(self, config: Dict):
        self.config = config

    def run(self):

        # Main run's components
        env_name = self.config["env"]["name"]
        agent_species_name = self.config["agents"]["name"]
        model_name = self.config["model"]["name"]

        # Hyperparameters
        n_timesteps: int = self.config["n_timesteps"]

        # Logging
        do_wandb: bool = self.config["do_wandb"]
        do_tb: bool = self.config["do_tb"]
        do_cli: bool = self.config["do_cli"]
        do_csv: bool = self.config["do_csv"]
        do_tqdm: bool = self.config["do_tqdm"]
        do_render: bool = self.config["do_render"]
        do_global_log: bool = self.config["do_global_log"]

        # Seed
        seed = try_get_seed(self.config)
        print(f"Using seed: {seed}")
        np.random.seed(seed)
        key_random = random.PRNGKey(seed)

        # ================ Initialization ================

        # Initialize loggers
        print(f"Current working directory: {os.getcwd()}")
        run_name = f"[{agent_species_name}_{model_name}_{env_name}]_{datetime.datetime.now().strftime('%dth%mmo_%Hh%Mmin%Ss')}_seed{seed}"
        if not do_global_log:
            dir_videos = f"./logs/videos/{run_name}"
            dir_metrics = f"./logs/{run_name}"
        else:
            dir_videos = "./logs/videos"
            dir_metrics = "./logs"
        print(f"\nStarting run {run_name}")

        list_loggers: List[Type[BaseLogger]] = []
        if do_wandb:
            list_loggers.append(
                LoggerWandB(
                    name_run=run_name,
                    config_run=self.config["wandb_config"],
                    **self.config["wandb_config"],
                )
            )
        if do_tb:
            list_loggers.append(LoggerTensorboard(log_dir=f"tensorboard/{run_name}"))
        if do_cli:
            list_loggers.append(LoggerCLI())
        if do_csv:
            list_loggers.append(LoggerCSV(dir_metrics=dir_metrics, do_log_phylo_tree=False))

        # Create the env
        EnvClass = env_name_to_EnvClass[env_name]
        self.config["env"]["dir_videos"] = dir_videos
        env = EnvClass(
            config=self.config["env"],
            n_agents_max=self.config["n_agents_max"],
            n_agents_initial=self.config["n_agents_initial"],
        )
        observation_space_dict = env.get_observation_space_dict()
        action_space_dict = env.get_action_space_dict()
        observation_class = env.get_class_observation_agent()
        action_class = env.get_class_action_agent()

        # Create the model
        ModelClass = model_name_to_ModelClass[model_name]
        model = ModelClass(
            observation_space_dict=observation_space_dict,
            action_space_dict=action_space_dict,
            observation_class=observation_class,
            action_class=action_class,
            **self.config["model"],
        )

        # Create the agent's species
        AgentSpeciesClass = agent_name_to_AgentSpeciesClass[agent_species_name]
        agent_species = AgentSpeciesClass(
            config=self.config["agents"],
            n_agents_max=self.config["n_agents_max"],
            n_agents_initial=self.config["n_agents_initial"],
            model=model,
        )

        # =============== Start simulation ===============
        print("Starting simulation...")
        key_random, subkey = random.split(key_random)
        (
            observations_agents,
            dict_reproduction,
            list_deaths,
            done_env,
            info_env,
        ) = env.reset(key_random=subkey)

        # Log the metrics
        metrics: Dict[str, Any] = info_env.get("metrics", {})
        metrics_scalar, metrics_histogram = get_dicts_metrics(metrics)
        for logger in list_loggers:
            logger.log_scalars(metrics_scalar, timestep=0)
            logger.log_histograms(metrics_histogram, timestep=0)
            logger.log_eco_metrics(dict_reproduction, list_deaths, timestep=0)

        print("Starting agents...")
        key_random, subkey = random.split(key_random)
        agent_species.init(key_random=subkey)

        # ============== Simulation loop ===============
        print("Simulation started.")
        # Training loop
        for timestep_run in tqdm(range(n_timesteps), disable=not do_tqdm):

            # Render the environment
            if do_render:
                env.render()

            # Agents step
            key_random, subkey = random.split(key_random)
            actions = agent_species.react(
                key_random=subkey,
                batch_observations=observations_agents,
                dict_reproduction=dict_reproduction,
            )

            # Env step
            key_random, subkey = random.split(key_random)
            (
                observations_agents,
                dict_reproduction,
                list_deaths,
                done_env,
                info_env,
            ) = env.step(
                key_random=subkey,
                actions=actions,
            )

            # Log the metrics
            metrics: Dict[str, Any] = info_env["metrics"]
            metrics_scalar, metrics_histogram = get_dicts_metrics(metrics)
            for logger in list_loggers:
                logger.log_scalars(metrics_scalar, timestep_run)
                logger.log_histograms(metrics_histogram, timestep_run)
                logger.log_eco_metrics(dict_reproduction, list_deaths, timestep_run)

            # Finish the loop if the environment is done
            if done_env:
                print("Environment done.")
                break

        # Finish the WandB run.
        for logger in list_loggers:
            logger.close()


if __name__ == "__main__":
    main()
