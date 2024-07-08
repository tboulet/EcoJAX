# Logging
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

    # Initialize loggers
    run_name = config.get(
        "run_name", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    print(f"\nStarting run {run_name}")
    if not do_global_log:
        dir_metrics = f"./logs/{run_name}"
    else:
        dir_metrics = "./logs"

    list_loggers: List[Type[BaseLogger]] = []
    if do_wandb:
        list_loggers.append(
            LoggerWandB(
                name_run=run_name,
                config_run=config,
                **config["wandb_config"],
            )
        )
    if do_tb:
        list_loggers.append(LoggerTensorboard(log_dir=f"tensorboard/{run_name}"))
    if do_cli:
        list_loggers.append(LoggerCLI())
    if do_csv:
        list_loggers.append(LoggerCSV(dir_metrics=dir_metrics, do_log_phylo_tree=False))
    if do_tqdm:
        list_loggers.append(LoggerTQDM(n_timesteps=n_timesteps))
    if do_snakeviz:
        list_loggers.append(LoggerSnakeviz())
    if do_jax_prof:
        list_loggers.append(LoggerJaxProfiling())

    def render_eco_loop(x: Tuple[StateGlobal, Dict[str, Any]]) -> jnp.ndarray:

        global_state, info = x
        t = global_state.timestep_run.item()

        # Render the environment
        if do_render:
            env.render(state=global_state.state_env)

        # Log the metrics
        metrics_global: dict = info.get("metrics", {}).copy()
        metrics_global.update(get_runtime_metrics())
        metrics_scalar, metrics_histogram = get_dict_metrics_by_type(metrics_global)
        for logger in list_loggers:
            logger.log_scalars(metrics_scalar, t)
            logger.log_histograms(metrics_histogram, t)
            logger.log_eco_metrics(global_state.eco_information, t)

    def step_eco_loop(x: Tuple[StateGlobal, Dict[str, Any]]) -> jnp.ndarray:
        print("Running step_eco_loop...")
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

    def do_continue_eco_loop(x: Tuple[StateGlobal, Dict[str, Any]]) -> jnp.ndarray:
        global_state, info = x
        return jax.numpy.logical_and(
            ~global_state.done,
            global_state.timestep_run < n_timesteps,
        )

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
    for logger in list_loggers:
        logger.log_scalars(metrics_scalar, timestep=0)
        logger.log_histograms(metrics_histogram, timestep=0)
        logger.log_eco_metrics(eco_information, timestep=0)
    info = {"metrics": metrics_global}

    # Run the simulation
    print("Running the simulation...")
    
    global_state = StateGlobal(
        state_env=state_env,
        state_species=state_species,
        observations=observations,
        eco_information=eco_information,
        timestep_run=jnp.array(0),
        done=done,
        key_random=subkey,
    )
    
    # Do (some?) first step(s) to get global_state and info at the right structure
    for _ in range(1):
        with RuntimeMeter("warmup steps"):
            if do_continue_eco_loop((global_state, info)):
                global_state, info = step_eco_loop((global_state, info))

    # JIT after first steps
    step_eco_loop = jax.jit(step_eco_loop)
    do_continue_eco_loop = jax.jit(do_continue_eco_loop)
    
    # @jax.jit # only works with while_loop, scan, and fori_loop
    def do_n_steps(global_state, info):
        # Method : native for loop (apparently the fastest method for this case)
        for _ in range(period_eval):
            global_state, info = step_eco_loop((global_state, info))
        return global_state, info

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

    while do_continue_eco_loop((global_state, info)):
        # Render every period_eval steps
        with RuntimeMeter("render"):
            render_eco_loop((global_state, info))
        # Run period_eval steps
        with RuntimeMeter("step", n_calls=period_eval):
            global_state, info = do_n_steps(global_state, info)
    # Final render
    with RuntimeMeter("render"):
        render_eco_loop((global_state, info))
        # jax.debug.callback(render_eco_loop, (global_state, info))

    print("End of simulation")
    print(f"Total runtime: {RuntimeMeter.get_total_runtime()}s")
    print("Stage runtimes:")
    pprint(RuntimeMeter.get_runtimes())
    print("Average stage runtimes:")
    pprint(RuntimeMeter.get_average_runtimes())

    # Close the loggers
    for logger in list_loggers:
        logger.close()
