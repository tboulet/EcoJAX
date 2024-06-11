# Logging
import os
import wandb
from tensorboardX import SummaryWriter
import csv


# Config system
import hydra
from omegaconf import OmegaConf, DictConfig
from ecojax.register_hydra import register_hydra_resolvers

register_hydra_resolvers()

# Utils
from tqdm import tqdm
import datetime
from time import time, sleep
from typing import Dict, Type
import cProfile

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
        os.makedirs(dir_videos, exist_ok=True)
        os.makedirs(f"logs/metrics", exist_ok=True)
    else:
        dir_videos = "logs/videos"
        path_csv = "logs/metrics/metrics.csv"
        os.makedirs(f"logs/videos", exist_ok=True)
        os.makedirs(f"logs/metrics", exist_ok=True)
    print(f"\nStarting run {run_name}")
    if do_snakeviz:
        pr = cProfile.Profile()
        pr.enable()
    if do_wandb:
        run = wandb.init(
            name=run_name,
            config=config,
            **config["wandb_config"],
        )
    if do_tb:
        tb_writer = SummaryWriter(log_dir=f"tensorboard/{run_name}")
    if do_csv:
        file_csv = open(path_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(file_csv)
        
        

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
            metrics: Dict[str, jax.Array] = info_env["metrics"]
            if do_wandb:
                wandb.log(metrics, step=timestep_run)  # TODO : check if that works
            if do_tb:
                for metric_name, metric_value in metrics.items():
                    if is_scalar(metric_value):
                        tb_writer.add_scalar(metric_name, metric_value, timestep_run)
                    elif is_array(metric_value):
                        metric_value = metric_value[~np.isnan(metric_value)]
                        if len(metric_value) > 0:
                            tb_writer.add_histogram(
                                metric_name, metric_value, timestep_run
                            )
                    else:
                        raise NotImplementedError
            if do_cli:
                print(f"Metrics at step {timestep_run}:\n{metrics}")
            if do_csv:
                for metric_name, metric_value in metrics.items():
                    if is_scalar(metric_value):
                        csv_writer.writerow(
                            [
                                timestep_run,
                                metric_name,
                                "",
                                float(metric_value),
                            ]
                        )
                    elif is_array(metric_value):
                        for agent_id in range(len(metric_value)):
                            csv_writer.writerow(
                                [
                                    timestep_run,
                                    metric_name,
                                    agent_id,
                                    float(metric_value[agent_id]),
                                ]
                            )
                    else:
                        raise NotImplementedError

        # Finish the loop if the environment is done
        if done_env:
            print("Environment done.")
            break

    # Finish the WandB run.
    if do_wandb:
        run.finish()
    if do_tb:
        tb_writer.close()
    if do_cli:
        print("Simulation done.")
    if do_csv:
        file_csv.close()
    if do_snakeviz:
        pr.disable()
        pr.dump_stats("logs/profile_stats.prof")
        print("Profile stats dumped to logs/profile_stats.prof")


if __name__ == "__main__":
    main()
