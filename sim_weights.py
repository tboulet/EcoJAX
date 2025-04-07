# Logging
import os
import cProfile
import pickle
from ecojax.agents.adaptive_rl import AdaptiveRL_AgentSpecies
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
from ecojax.types import ActionAgent, ObservationAgent, StateEnv, StateGlobal, StateSpecies

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
        
        agent_config_path = input("Enter the path to the agent config file: ")
        # if agent_config_path:
        #     with open(agent_config_path, "r") as f:
        #         self.config["agents"] = OmegaConf.load(f)
        # else:
        #     print("No agent config file provided, using default config.")
            
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
        self.config["agents"]["run_name"] = run_name
        self.config["env"]["run_name"] = run_name
        
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
        jnp.array
        observation_space = env.get_observation_space()
        action_space = env.get_action_space()

        # Create the model
        ModelClass = model_name_to_ModelClass[model_name]

        # Create the agent's species
        AgentSpeciesClass = agent_name_to_AgentSpeciesClass[agent_species_name]
        agent_species : AdaptiveRL_AgentSpecies = AgentSpeciesClass(
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
        
        list_agent_weights_np = []
        while True:
            try:
                command = input("Enter a command: ")
                
                if command.startswith("load"):
                    _, path = command.split()
                    
                    for agent_weight_name in os.listdir(path):
                        with open(os.path.join(path, agent_weight_name), "rb") as f:
                            agent_weights = pickle.load(f)
                            list_agent_weights_np.append(agent_weights)
                    print(f"Loaded {len(list_agent_weights_np)} agent weights from {path}")
                    
                elif command == "done":
                    break
                
            except Exception as e:
                print(f"Error: {e}. Please try again.")
                
        def obs_to_dict_action(obs : ObservationAgent, idx = 0) -> Dict[str, ActionAgent]:
            """Convert the observation to a dict of actions representing probs"""
            try:
                # Get the jnp weights
                params_np = list_agent_weights_np[idx]
                params = jax.tree_util.tree_map(lambda x: jnp.array(x), params_np)
                # Define the action space
                key_random = random.PRNGKey(seed)
                logits = agent_species.model.apply(variables={"params": params}, x=obs, key_random=key_random)
                probs = jax.nn.softmax(logits)
                action_to_probs = {name_action : probs[idx_action] for idx_action, name_action in env.action_idx_to_meaning().items()}
            except Exception as e:
                print(f"Error: {e}")
                breakpoint()
        
            return action_to_probs
        
        # Test the function
        obs = observation_space.sample(jax.random.PRNGKey(seed))
        obs["table_value_fruits"] = jnp.array([0, 0, 0, 0])
        action = obs_to_dict_action(obs)
        print(f"Action: {action}")
        
                
if __name__ == "__main__":
    main()
