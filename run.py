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
    config_dirs_to_log: Dict[str, bool] = config["config_dirs_to_log"]

    # # Video recording
    # do_video: bool = config["do_video"]
    # n_steps_between_videos: int = config["n_steps_between_videos"]
    # n_steps_per_video: int = config["n_steps_per_video"]
    # n_steps_between_frames: int = config["n_steps_between_frames"]
    # assert (
    #     n_steps_per_video <= n_steps_between_videos
    # ) or not do_video, "len_video must be less than or equal to freq_video"

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

    # Create the agent's species
    agent_species_name: str = config["agents"]["name"]
    AgentSpeciesClass = agent_name_to_AgentSpeciesClass[agent_species_name]
    agent_species = AgentSpeciesClass(
        config=config["agents"],
        n_agents_max=config["n_agents_max"],
        n_agents_initial=config["n_agents_initial"],
        observation_space_dict=observation_space_dict,
        action_space_dict=action_space_dict,
        observation_class=observation_class,
        action_class=action_class,
    )

    # Initialize loggers
    run_name = f"[{env_name}]_{datetime.datetime.now().strftime('%dth%mmo_%Hh%Mmin%Ss')}_seed{seed}"
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
        state_env,
        observations_agents,
        dict_reproduction,
        done_env,
        info_env,
    ) = env.reset(key_random=subkey)

    # ============== Simulation loop ===============
    print("Simulation started.")
    # Training loop
    for t in tqdm(range(n_timesteps), disable=not do_tqdm):

        # Render the environment
        env.render(state=state_env)

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
            state_env,
            observations_agents,
            dict_reproduction,
            done_env,
            info_env,
        ) = env.step(
            key_random=subkey,
            state=state_env,
            actions=actions,
        )
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
