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
import random
import numpy as np

# Project imports
from src.time_measure import RuntimeMeter
from src.utils import try_get_seed
from src.environment import env_name_to_EnvClass
from src.agents import agent_name_to_AgentSpeciesClass
from src.models import model_name_to_ModelClass

@hydra.main(config_path="configs", config_name="default.yaml")
def main(config: DictConfig):
    print("Configuration used :")
    print(OmegaConf.to_yaml(config))
    config = OmegaConf.to_container(config, resolve=True)
    
    # ================ Configuration ================

    # Get configuration parameters
    n_timesteps: int = config["n_timesteps"]
    do_wandb: bool = config["do_wandb"]
    do_tb: bool = config["do_tb"]
    do_cli: bool = config["do_cli"]
    do_tqdm: bool = config["do_tqdm"]
    do_snakeviz: bool = config["do_snakeviz"]

    # Set the seeds
    seed = try_get_seed(config)
    random.seed(seed)
    np.random.seed(seed)
    print(f"Using seed: {seed}")

    # Create the env
    print("Creating the env...")
    env_name: str = config["env"]["name"]
    EnvClass = env_name_to_EnvClass[env_name]
    env = EnvClass(config["env"])

    

    # Initialize loggers
    run_name = f"[{None}]_[{env_name}]_{datetime.datetime.now().strftime('%dth%mmo_%Hh%Mmin%Ss')}_seed{seed}"
    os.makedirs("logs", exist_ok=True)
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

    # Training loop
    for t in tqdm(range(n_timesteps), disable=not do_tqdm):
        raise
    for iteration in tqdm(range(n_iterations), disable=not do_tqdm):
        # Get the solver result, and measure the time.
        with RuntimeMeter("solver") as rm:
            y_pred = solver.fit(x_data=x_data)
            sleep(0.1)  # Simulate a long computation time
        
        # Compute metrics
        with RuntimeMeter("metric") as rm:
            metric_result = dict()
            # Compute MSE
            y_data = env.get_labels()
            metric_result["mse"] = ((y_data - y_pred) ** 2).mean()
            # Utils metric : runtimes of each stage in the pipeline and iteration number
            for stage_name, stage_runtime in rm.get_stage_runtimes().items():
                metric_result[f"runtime_{stage_name}"] = stage_runtime
            metric_result["total_runtime"] = rm.get_total_runtime()
            metric_result["iteration"] = iteration

        # Log the metrics
        with RuntimeMeter("log") as rm:
            if do_wandb:
                cumulative_solver_time_in_ms = int(
                    rm.get_stage_runtime("solver") * 1000
                )
                wandb.log(metric_result, step=cumulative_solver_time_in_ms)
            if do_tb:
                for metric_name, metric_result in metric_result.items():
                    tb_writer.add_scalar(
                        f"metrics/{metric_name}",
                        metric_result,
                        global_step=rm.get_stage_runtime("solver"),
                    )
            if do_cli:
                print(
                    f"Metric results at iteration {iteration} for metric {metric_name}: {metric_result}"
                )

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
