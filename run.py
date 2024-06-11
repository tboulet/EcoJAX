# Logging
import os
from ecojax.loggers import BaseLogger
from ecojax.loggers.cli import LoggerCLI
from ecojax.loggers.csv import LoggerCSV
from ecojax.loggers.snakeviz import LoggerSnakeviz
from ecojax.loggers.tensorboard import LoggerTensorboard
from ecojax.loggers.wandb import LoggerWandB


# Config system
import hydra
from omegaconf import OmegaConf, DictConfig
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
from ecojax.utils import is_array, is_scalar, try_get_seed


@hydra.main(config_path="configs", config_name="default.yaml")
def main(config: DictConfig):
    print("Configuration used :")
    print(OmegaConf.to_yaml(config))
    config = OmegaConf.to_container(config, resolve=True)

    # ================ Configuration ================

    # Main run's components
    env_name = config["env"]["name"]
    agent_species_name = config["agents"]["name"]
    model_name = config["model"]["name"]

    # Hyperparameters
    n_timesteps: int = config["n_timesteps"]

    # Logging
    do_wandb: bool = config["do_wandb"]
    do_tb: bool = config["do_tb"]
    do_cli: bool = config["do_cli"]
    do_csv: bool = config["do_csv"]
    do_tqdm: bool = config["do_tqdm"]
    do_snakeviz: bool = config["do_snakeviz"]
    do_render: bool = config["do_render"]
    do_global_log: bool = config["do_global_log"]

    # Seed
    seed = try_get_seed(config)
    print(f"Using seed: {seed}")
    np.random.seed(seed)
    key_random = random.PRNGKey(seed)

    # ================ Initialization ================

    # Initialize loggers
    print(f"Current working directory: {os.getcwd()}")
    run_name = f"[{agent_species_name}_{model_name}_{env_name}]_{datetime.datetime.now().strftime('%dth%mmo_%Hh%Mmin%Ss')}_seed{seed}"
    if not do_global_log:
        dir_videos = f"logs/videos/{run_name}"
        path_csv = f"logs/metrics/{run_name}.csv"
    else:
        dir_videos = "logs/videos"
        path_csv = "logs/metrics/metrics.csv"
    print(f"\nStarting run {run_name}")

    list_loggers : List[Type[BaseLogger]] = []
    if do_wandb:
        list_loggers.append(
            LoggerWandB(
                name_run=run_name,
                config_run=config["wandb_config"],
                **config["wandb_config"],
            )
        )
    if do_tb:
        list_loggers.append(LoggerTensorboard(log_dir=f"tensorboard/{run_name}"))
    if do_cli:
        list_loggers.append(LoggerCLI())
    if do_csv:
        list_loggers.append(LoggerCSV(path_csv=path_csv))
    if do_snakeviz:
        list_loggers.append(LoggerSnakeviz())

    # Create the env
    EnvClass = env_name_to_EnvClass[env_name]
    config["env"]["dir_videos"] = dir_videos
    env = EnvClass(
        config=config["env"],
        n_agents_max=config["n_agents_max"],
        n_agents_initial=config["n_agents_initial"],
    )
    observation_space_dict = env.get_observation_space_dict()
    action_space_dict = env.get_action_space_dict()
    observation_class = env.get_class_observation_agent()
    action_class = env.get_class_action_agent()

    # Create the model
    ModelClass = model_name_to_ModelClass[model_name]
    model = ModelClass(
        config=config["model"],
        observation_space_dict=observation_space_dict,
        action_space_dict=action_space_dict,
        observation_class=observation_class,
        action_class=action_class,
    )

    # Create the agent's species
    AgentSpeciesClass = agent_name_to_AgentSpeciesClass[agent_species_name]
    agent_species = AgentSpeciesClass(
        config=config["agents"],
        n_agents_max=config["n_agents_max"],
        n_agents_initial=config["n_agents_initial"],
        model=model,
    )

    # =============== Start simulation ===============
    print("Starting simulation...")
    key_random, subkey = random.split(key_random)
    (
        observations_agents,
        dict_reproduction,
        done_env,
        info_env,
    ) = env.reset(key_random=subkey)

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
            done_env,
            info_env,
        ) = env.step(
            key_random=subkey,
            actions=actions,
        )

        # Log the metrics
        if timestep_run % 100 == 0:
            metrics: Dict[str, Any] = info_env["metrics"]
            metric_scalar = {}
            metric_histogram = {}
            for key, value in metrics.items():
                if is_scalar(value):
                    metric_scalar[key] = value
                elif is_array(value):
                    value_non_nan = value[~np.isnan(value)]
                    if len(value_non_nan) > 0:
                        metric_histogram[key] = value_non_nan
            for logger in list_loggers:
                logger.log_scalars(metric_scalar, timestep_run)
                logger.log_histograms(metric_histogram, timestep_run)

        # Finish the loop if the environment is done
        if done_env:
            print("Environment done.")
            break

    # Finish the WandB run.
    for logger in list_loggers:
        logger.close()


if __name__ == "__main__":
    main()
