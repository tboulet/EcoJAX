# Logging
from collections import defaultdict
import os
import cProfile

import jax.experimental
from ecojax.agents.neuro_evolution import AgentNE, NeuroEvolutionAgentSpecies
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
    n_agents_max = config["n_agents_max"]
    period_video: int = int(max(1, config["period_video"]))
    t_last_video: int = -period_video

    print("period_video:", period_video)

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

    do_advanced_logging = False
    enhanced_logging_start = n_timesteps - 50000
    do_eval = False
    eval_start = n_timesteps - 5000
    log_metrics_interval = 500
    eval_interval = 100
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

    def step_eco_loop(x: Tuple[StateGlobal, Dict[str, Any]]) -> Tuple[StateGlobal, Dict[str, Any]]:
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
            info
        )

    def run_feed_selectivity_eval(key_random, agents_state, curr_obss) -> jnp.ndarray:
        def get_feed_prob(agent: AgentNE, obs) -> jnp.ndarray:
            logits = agent_species.model.apply(
                variables={"params": agent.params},
                x=obs,
                key_random=key_random,
            )
            return jax.nn.softmax(logits)[5]

        def get_probs(vis_field):
            return jax.vmap(
                get_feed_prob, in_axes=(0, None))(agents_state.agents, {"visual_field": vis_field}
            ).reshape((1, n_agents_max))

        # sample 10 random agents' visual fields
        vis_fields = random.choice(key_random, curr_obss["visual_field"], shape=(10,), replace=False)

        # retain the 'plants' channel of the visual field, set the rest to 0 except for the agent's own position
        vis_fields = jnp.concatenate([vis_fields[:, :1], jnp.zeros_like(vis_fields[:, 1:])], axis=1)
        c = (vis_fields.shape[1] - 1) // 2
        vis_fields = vis_fields.at[:, 1, c, c].set(1)

        # for each of the sampled visual fields, generate 5 variants:
        #     - baseline: no other agent present
        #     - facing other agent who is neither infant nor offspring
        #     - facing other agent who is infant but not offspring
        #     - facing other agent who is offspring but not infant
        #     - facing other agent who is both infant and offspring
        def process_visual_field(vis_field: jnp.ndarray) -> jnp.ndarray:
            results = [
                get_probs(vis_field),
                get_probs(vis_field.at[c - 1, c, 1].set(1)),
                get_probs(vis_field.at[c - 1, c, 1].set(1).at[c- 1, c, 2].set(1)),
                get_probs(vis_field.at[c - 1, c, 1].set(1).at[c - 1, c, 3].set(1)),
                get_probs(vis_field.at[c - 1, c, 1].set(1).at[c - 1, c, 2].set(1).at[c - 1, c, 3].set(1))
            ]
            return jnp.concatenate(results)

        return jnp.mean(jax.vmap(process_visual_field)(vis_fields), axis=0)

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

    def save_data(data_dict, filename, concat=True):
        if concat:
            data_dict = {k: jnp.concatenate(v) for k, v in data_dict.items()}
        fp = os.path.join(dir_metrics, filename)
        pd.DataFrame(data_dict).to_csv(fp, index=False)

    def flush_data(data_dict, filename):
        save_data(data_dict, filename)
        return defaultdict(list)

    # data dicts
    feeding_data = defaultdict(list)
    birth_data = defaultdict(list)
    death_data = defaultdict(list)
    metrics_data = defaultdict(list)

    def process_step_data_basic(t, info):
        if t % log_metrics_interval == 0:
            num_feed = jnp.sum(info["metrics"]["feeders"] > 0)
            prop_feed_offspring = jnp.sum(
                info["metrics"]["to_offspring"] > 0
            ) / jnp.maximum(1, num_feed)
            prop_face_offspring = info["metrics"]["num_facing_offspring"] / jnp.maximum(
                1, info["metrics"]["num_facing_agent"]
            )
            death_ages = info["metrics"]["life_expectancy"]
            avg_death_age = jnp.nanmean(death_ages)
            prop_dead_adults = (
                1
                + jnp.nanmean(jnp.sign(death_ages - config["env"]["infancy_duration"]))
            ) / 2

            metrics_data["timestep"].append(t)
            metrics_data["n_agents"].append(info["metrics"]["n_agents"])
            metrics_data["n_plants"].append(info["metrics"]["n_plants"])
            metrics_data["life_expectancy"].append(avg_death_age)
            metrics_data["survival_to_adulthood"].append(prop_dead_adults)
            metrics_data["num_feedings"].append(num_feed)
            metrics_data["prop_feed_offspring"].append(prop_feed_offspring)
            metrics_data["prop_face_offspring"].append(prop_face_offspring)

    def process_step_data_enhanced(t, global_state, info):
        # process feeding data
        feeders = jnp.where(info["metrics"]["feeders"] == 1)[0].astype(jnp.int32)
        feedees = info["metrics"]["feedees"][feeders].astype(jnp.int32)
        feeding_data["feeder"].append(feeders)
        feeding_data["feedee"].append(feedees)
        feeding_data["feeder_age"].append(
            info["metrics"]["age"][feeders].astype(jnp.int32)
        )
        feeding_data["feedee_age"].append(
            info["metrics"]["age"][feedees].astype(jnp.int32)
        )
        feeding_data["to_offspring"].append(
            info["metrics"]["to_offspring"][feeders].astype(jnp.int32)
        )
        feeding_data["timestep"].append(jnp.full_like(feeders, t))

        # process birth data
        newborns = jnp.where(global_state.eco_information.are_newborns_agents == 1)[
            0
        ].astype(jnp.int32)
        parents = global_state.eco_information.indexes_parents.reshape(
            (n_agents_max,)
        )[newborns]
        birth_data["agent"].append(newborns)
        birth_data["parent"].append(parents)
        birth_data["timestep"].append(jnp.full_like(newborns, t))

        # process death data
        life_exp = info["metrics"]["life_expectancy"].astype(jnp.int32)
        dead_agents = jnp.where(life_exp > 0)[0].astype(jnp.int32)
        death_data["agent"].append(dead_agents)
        death_data["age"].append(life_exp[dead_agents])
        death_data["timestep"].append(jnp.full_like(dead_agents, t))

    # Do (some?) first step(s) to get global_state and info at the right structure
    for t in range(1):
        global_state, info = step_eco_loop((global_state, info))
        process_step_data_basic(t, info)

    # JIT after first steps
    # step_eco_loop = jax.jit(step_eco_loop)
    run_feed_selectivity_eval = jax.jit(run_feed_selectivity_eval)

    t = int(global_state.timestep_run)
    pbar = tqdm(total=n_timesteps, desc="Running simulation", initial=t)
    while global_state.timestep_run < n_timesteps:
        t = global_state.timestep_run.item()
        if do_render and t - t_last_video >= period_video:
            env.render(state=global_state.state_env)
            t_last_video = t

        global_state, info = step_eco_loop((global_state, info))

        # record some metrics and event data
        process_step_data_basic(t, info)
        if do_advanced_logging and t >= enhanced_logging_start:
            process_step_data_enhanced(t, global_state, info)

        if do_eval and t >= eval_start and t % eval_interval == 0:
            eval_results = run_feed_selectivity_eval(
                random.PRNGKey(t),
                global_state.state_species,
                global_state.observations
            )
            jnp.save(
                os.path.join(dir_metrics, f"eval_results_{t}.npy"),
                eval_results
            )

        if t % flush_interval == 0 and t >= flush_interval:
            tqdm.write(f"Saving data at timestep {t}...")
            if do_advanced_logging and t >= enhanced_logging_start:
                feeding_data = flush_data(feeding_data, f"feeding_data_{t}.csv")
                birth_data = flush_data(birth_data, f"birth_data_{t}.csv")
                death_data = flush_data(death_data, f"death_data_{t}.csv")
            save_data(metrics_data, "metrics_data.csv", concat=False)

        t_new = int(global_state.timestep_run)
        pbar.update(t_new - t)
        t = t_new

    print("End of simulation")

    # # Close the loggers
    # for logger in list_loggers:
    #     logger.close()
