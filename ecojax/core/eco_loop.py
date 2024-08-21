# Logging
from collections import defaultdict
import os
import cProfile

import jax.experimental
from ecojax.loggers import BaseLogger
from ecojax.loggers.cli import LoggerCLI
from ecojax.loggers.csv import LoggerCSV
from ecojax.loggers.jax_profiling import LoggerJaxProfiling
from ecojax.loggers.snakeviz import LoggerSnakeviz
from ecojax.loggers.tensorboard import LoggerTensorboard
from ecojax.loggers.tqdm import LoggerTQDM
from ecojax.loggers.wandb import LoggerWandB

# Utils
from tqdm import tqdm
import datetime
from time import time, sleep
from typing import Any, Dict, List, Tuple, Type
from pprint import pprint

# ML libraries
import pandas as pd
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax.lax import while_loop

# Project imports
from ecojax.environment import EcoEnvironment, env_name_to_EnvClass
from ecojax.agents import AgentSpecies, agent_name_to_AgentSpeciesClass
from ecojax.metrics.utils import get_dict_metrics_by_type
from ecojax.models import model_name_to_ModelClass
from ecojax.core.eco_info import EcoInformation
from ecojax.time_measure import RuntimeMeter, get_runtime_metrics
from ecojax.types import ObservationAgent, StateEnv, StateGlobal, StateSpecies
from ecojax.utils import check_jax_device, is_array, is_scalar, try_get_seed


def eco_loop(
    env: EcoEnvironment,
    agent_species: AgentSpecies,
    config: Dict,
    key_random: jnp.ndarray,
):
    """
    Perform the main loop of the simulation.

    Args:
        env (EcoEnvironment): the environment
        agent_species (AgentSpecies): the species of agents
        config (Dict): the configuration
        key_random (jnp.ndarray): the random key
    """
    # Hyperparameters
    n_timesteps: int = config["n_timesteps"]
    period_eval: int = int(max(1, config["period_eval"]))
    period_video: int = int(max(1, config["period_video"]))
    t_last_video: int = -period_video

    # Logging
    do_wandb: bool = config["do_wandb"]
    do_tb: bool = config["do_tb"]
    do_cli: bool = config["do_cli"]
    do_csv: bool = config["do_csv"]
    do_tqdm: bool = config["do_tqdm"]
    do_snakeviz: bool = config["do_snakeviz"]
    do_jax_prof: bool = config.get("do_jax_prof", False)
    do_render: bool = config["do_render"]
    do_global_log: bool = config["do_global_log"]
    dir_metrics: str = config["log_dir_path"]
    run_name: str = config["run_name"]

    enhanced_logging_start = n_timesteps - 50000
    log_metrics_interval = 500
    flush_interval = 10000

    print(f"\nStarting run {run_name}")

    # Initialize loggers
    # list_loggers: List[Type[BaseLogger]] = []
    # if do_wandb:
    #     list_loggers.append(
    #         LoggerWandB(
    #             name_run=run_name,
    #             config_run=config,
    #             **config["wandb_config"],
    #         )
    #     )
    # if do_tb:
    #     list_loggers.append(LoggerTensorboard(log_dir=f"tensorboard/{run_name}"))
    # if do_cli:
    #     list_loggers.append(LoggerCLI())
    # if do_csv:
    #     list_loggers.append(LoggerCSV(dir_metrics=dir_metrics, do_log_phylo_tree=False))
    # if do_tqdm:
    #     list_loggers.append(LoggerTQDM(n_timesteps=n_timesteps))
    # if do_snakeviz:
    #     list_loggers.append(LoggerSnakeviz())
    # if do_jax_prof:
    #     list_loggers.append(LoggerJaxProfiling())

    # def log_and_render_eco_loop(x: Tuple[StateGlobal, Dict[str, Any], int]) -> int:
    #     global_state, info, t_last_video = x
    #     t = global_state.timestep_run.item()

    #     # Render the environment
    #     if do_render and t - t_last_video >= period_video:
    #         env.render(state=global_state.state_env)
    #         t_last_video = t

    #     # Log the metrics
    #     metrics_global: dict = info.get("metrics", {}).copy()
    #     metrics_global.update(get_runtime_metrics())
    #     metrics_scalar, metrics_histogram = get_dict_metrics_by_type(metrics_global)
    #     for logger in list_loggers:
    #         logger.log_scalars(metrics_scalar, t)
    #         logger.log_histograms(metrics_histogram, t)
    #         logger.log_eco_metrics(global_state.eco_information, t)

    #     return t_last_video

    def step_eco_loop(x: Tuple[StateGlobal, Dict[str, Any]]) -> jnp.ndarray:
        # print("Running step_eco_loop...")
        global_state, info = x
        key_random = global_state.key_random

        # Agents step
        key_random, subkey = random.split(key_random)
        new_state_species, actions, info_species = agent_species.react(
            state=global_state.state_species,
            batch_observations=global_state.observations,
            eco_information=global_state.eco_information,
            key_random=subkey,
        )

        # Env step
        key_random, subkey = random.split(key_random)
        new_state_env, new_observations, new_eco_information, new_done, info_env = (
            env.step(
                state=global_state.state_env,
                actions=actions,
                key_random=subkey,
                state_species=new_state_species,  # optional, to allow the environment to access the state of the species
            )
        )

        # Return the new global state
        key_random, subkey = random.split(key_random)
        metrics_env = info_env.get("metrics", {})
        metrics_species = info_species.get("metrics", {})
        metrics_global = {**metrics_env, **metrics_species}
        info = {"metrics": metrics_global}
        return (
            StateGlobal(
                state_env=new_state_env,
                state_species=new_state_species,
                observations=new_observations,
                eco_information=new_eco_information,
                timestep_run=global_state.timestep_run + 1,
                done=new_done,
                key_random=subkey,
            ),
            info,
        )

    # def do_continue_eco_loop(x: Tuple[StateGlobal, Dict[str, Any]]) -> jnp.ndarray:
    #     global_state, info = x
    #     return jax.numpy.logical_and(
    #         ~global_state.done,
    #         global_state.timestep_run < n_timesteps,
    #     )

    # Initialize the environment
    print("Initializing environment...")
    key_random, subkey = random.split(key_random)
    (
        state_env,
        observations,
        eco_information,
        done,
        info_env,
    ) = env.reset(key_random=subkey)

    # Initialize the species
    print("Initializing agents...")
    key_random, subkey = random.split(key_random)
    state_species = agent_species.reset(key_random=subkey)
    info_species = {"metrics": {}}

    # Initialize the metrics and log them
    print("Initializing metrics...")
    metrics_env = info_env.get("metrics", {})
    metrics_species = info_species.get("metrics", {})
    metrics_global = {**metrics_env, **metrics_species}
    metrics_scalar, metrics_histogram = get_dict_metrics_by_type(metrics_global)
    # for logger in list_loggers:
    #     logger.log_scalars(metrics_scalar, timestep=0)
    #     logger.log_histograms(metrics_histogram, timestep=0)
    #     logger.log_eco_metrics(eco_information, timestep=0)
    info = {"metrics": metrics_global}

    global_state = StateGlobal(
        state_env=state_env,
        state_species=state_species,
        observations=observations,
        eco_information=eco_information,
        timestep_run=jnp.array(0),
        done=done,
        key_random=subkey,
    )

    def create_data_dict(keys):
        return {key: [] for key in keys}

    def save_data(data_dict, filename, concat=True):
        if concat:
            data_dict = {
                k: jnp.concatenate(v) for k, v in data_dict.items()
            }
        fp = os.path.join(dir_metrics, filename)
        pd.DataFrame(data_dict).to_csv(fp, index=False)

    def flush_data(data_dict, filename):
        save_data(data_dict, filename)
        return create_data_dict(data_dict.keys())

    # data dicts
    feeding_data = defaultdict(list)
    birth_data = defaultdict(list)
    death_data = defaultdict(list)
    metrics_data = defaultdict(list)

    def process_step_data_basic(t, info):
        if t % log_metrics_interval:
            num_feed = jnp.sum(info["metrics"]["feeders"] > 0)
            prop_feed_offspring = jnp.sum(
                info["metrics"]["to_offspring"] > 0
            ) / jnp.maximum(1, num_feed)
            prop_face_offspring = info["metrics"]["num_facing_offspring"] / jnp.maximum(
                1, info["metrics"]["num_facing_agent"]
            )
            life_exp = jnp.nanmean(info["metrics"]["life_expectancy"])
            metrics_data["timestep"].append(t)
            metrics_data["n_agents"].append(info["metrics"]["n_agents"])
            metrics_data["n_plants"].append(info["metrics"]["n_plants"])
            metrics_data["life_expectancy"].append(life_exp)
            metrics_data["num_feedings"].append(num_feed)
            metrics_data["prop_feed_offspring"].append(prop_feed_offspring)
            metrics_data["prop_face_offspring"].append(prop_face_offspring)


    def process_step_data_enhanced(t, global_state, info):
        # process feeding data
        feeders = jnp.where(info["metrics"]["feeders"] == 1)[0].astype(jnp.int32)
        # feeders = jnp.where(info["metrics"]["to_offspring"] == 1)[0].astype(jnp.int32)
        feedees = info["metrics"]["feedees"][feeders].astype(jnp.int32)
        feeding_data["feeder"].append(feeders)
        feeding_data["feedee"].append(feedees)
        feeding_data["feeder_age"].append(info["metrics"]["age"][feeders].astype(jnp.int32))
        feeding_data["feedee_age"].append(info["metrics"]["age"][feedees].astype(jnp.int32))
        feeding_data["to_offspring"].append(info["metrics"]["to_offspring"][feeders].astype(jnp.int32))
        feeding_data["timestep"].append(jnp.full_like(feeders, t))

        # process birth data
        newborns = jnp.where(global_state.eco_information.are_newborns_agents == 1)[
            0
        ].astype(jnp.int32)
        birth_data["agent"].append(newborns)
        birth_data["timestep"].append(jnp.full_like(newborns, t))

        # process death data
        life_exp = info["metrics"]["life_expectancy"].astype(jnp.int32)
        dead_agents = jnp.where(life_exp > 0)[0].astype(jnp.int32)
        death_data["agent"].append(dead_agents)
        death_data["age"].append(life_exp[dead_agents])
        death_data["timestep"].append(jnp.full_like(dead_agents, t))

    # Do (some?) first step(s) to get global_state and info at the right structure
    for t in range(1):
        with RuntimeMeter("warmup steps"):
            # if do_continue_eco_loop((global_state, info)):
            global_state, info = step_eco_loop((global_state, info))
            process_step_data_basic(t, info)

    # JIT after first steps
    step_eco_loop = jax.jit(step_eco_loop)
    # do_continue_eco_loop = jax.jit(do_continue_eco_loop)
    #

    t = int(global_state.timestep_run)
    pbar = tqdm(total=n_timesteps, desc="Running simulation", initial=t)
    while global_state.timestep_run < n_timesteps:
        global_state, info = step_eco_loop((global_state, info))

        process_step_data_basic(t, info)
        if t >= enhanced_logging_start:
            process_step_data_enhanced(t, global_state, info)

        if t % flush_interval == 0 and t >= flush_interval:
            tqdm.write(f"Saving data at timestep {t}...")
            if t >= enhanced_logging_start:
                feeding_data = flush_data(
                    feeding_data, f"feeding_data_{t}.csv"
                )
                birth_data = flush_data(birth_data, f"birth_data_{t}.csv")
                death_data = flush_data(death_data, f"death_data_{t}.csv")
            save_data(metrics_data, "metrics_data.csv", concat=False)

        t_new = int(global_state.timestep_run)
        pbar.update(t_new - t)
        t = t_new

    # @jax.jit # only works with while_loop, scan, and fori_loop
    # def do_n_steps(global_state, info):
    #     # Method : native for loop (apparently the fastest method for this case)
    #     for _ in range(period_eval):
    #         global_state, info = step_eco_loop((global_state, info))
    #     return global_state, info

    # # Method : native while loop
    # t = 0
    # while do_continue_eco_loop((global_state, info)) and t < period_eval:
    #     global_state, info = step_eco_loop((global_state, info))
    #     t += 1
    # return global_state, info

    # # Method : JAX's fori_loop
    # return jax.lax.fori_loop(
    #     0, period_eval, lambda i, x: step_eco_loop(x), (global_state, info)
    # )

    # # Method : JAX's scan
    # (global_state, info), elems = jax.lax.scan(f=lambda x, el: (step_eco_loop(x), None), init=(global_state, info), xs=None, length=period_eval)
    # return global_state, info

    # while do_continue_eco_loop((global_state, info)):
    #     # Render every period_eval steps
    #     with RuntimeMeter("render"):
    #         t_last_video = log_and_render_eco_loop((global_state, info, t_last_video))
    #     # Run period_eval steps
    #     with RuntimeMeter("step", n_calls=period_eval):
    #         global_state, info = do_n_steps(global_state, info)

    # # Final render
    # with RuntimeMeter("render"):
    #     t_last_video = log_and_render_eco_loop((global_state, info, t_last_video))
    #     # jax.debug.callback(log_and_render_eco_loop, (global_state, info))

    print("End of simulation")
    print(f"Total runtime: {RuntimeMeter.get_total_runtime()}s")
    print("Stage runtimes:")
    pprint(RuntimeMeter.get_runtimes())
    print("Average stage runtimes:")
    pprint(RuntimeMeter.get_average_runtimes())

    # # Close the loggers
    # for logger in list_loggers:
    #     logger.close()
