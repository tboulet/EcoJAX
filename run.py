# Logging
import os
import wandb
from tensorboardX import SummaryWriter

# Config system
import hydra
from omegaconf import OmegaConf, DictConfig

# Utils
from tqdm import tqdm
import datetime
from time import time, sleep
from typing import Dict, Type
import cProfile

# ML libraries
import numpy as np
from jax import random

# Project imports
from src.environment import env_name_to_EnvClass
from src.agents import agent_name_to_AgentSpeciesClass
from src.models import model_name_to_ModelClass
from src.video import VideoRecorder
from src.time_measure import RuntimeMeter
from src.utils import try_get_seed


@hydra.main(config_path="configs", config_name="default.yaml")
def main(config: DictConfig):
    print("Configuration used :")
    print(OmegaConf.to_yaml(config))
    config = OmegaConf.to_container(config, resolve=True)

    # ================ Configuration ================

    # Hyperparameters
    n_timesteps: int = config["n_timesteps"]

    # Logging
    do_wandb: bool = config["do_wandb"]
    do_tb: bool = config["do_tb"]
    do_cli: bool = config["do_cli"]
    do_tqdm: bool = config["do_tqdm"]
    do_snakeviz: bool = config["do_snakeviz"]
    do_render: bool = config["do_render"]
    config_dirs_to_log: Dict[str, bool] = config["config_dirs_to_log"]

    # ================ Initialization ================

    # Set the seeds
    seed = try_get_seed(config)
    print(f"Using seed: {seed}")
    np.random.seed(seed)
    key_random = random.PRNGKey(seed)

    # Create the env
    env_name: str = config["env"]["name"]
    EnvClass = env_name_to_EnvClass[env_name]
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
    model_name: str = config["model"]["name"]
    ModelClass = model_name_to_ModelClass[model_name]
    model = ModelClass(
        config=config["model"],
        observation_space_dict=observation_space_dict,
        action_space_dict=action_space_dict,
        observation_class=observation_class,
        action_class=action_class,
    )

    # Create the agent's species
    agent_species_name: str = config["agents"]["name"]
    AgentSpeciesClass = agent_name_to_AgentSpeciesClass[agent_species_name]
    agent_species = AgentSpeciesClass(
        config=config["agents"],
        n_agents_max=config["n_agents_max"],
        n_agents_initial=config["n_agents_initial"],
        model=model,
    )

    # Initialize loggers
    run_name = f"[{agent_species_name}_{model_name}_{env_name}]_{datetime.datetime.now().strftime('%dth%mmo_%Hh%Mmin%Ss')}_seed{seed}"
    os.makedirs(f"logs/runs/{run_name}", exist_ok=True)
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
        
        # Log the measures
        metrics = info_env["metrics"]
        
        # Finish the loop if the environment is done
        if done_env:
            print("Environment done.")
            break

    # Finish the WandB run.
    if do_wandb:
        run.finish()
    if do_tb:
        tb_writer.close()
    if do_snakeviz:
        pr.disable()
        pr.dump_stats("logs/profile_stats.prof")
        print("Profile stats dumped to logs/profile_stats.prof")


if __name__ == "__main__":
    main()
