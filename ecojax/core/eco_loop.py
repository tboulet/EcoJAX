from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Union
import jax.experimental
import numpy as np

import jax
from jax import random
import jax.numpy as jnp
from flax import struct
import flax.linen as nn
from jax.lax import while_loop

from ecojax.agents.base_agent_species import BaseAgentSpecies
from ecojax.environment.base_env import BaseEcoEnvironment
from ecojax.models.base_model import BaseModel
from ecojax.types import ObservationAgent, ActionAgent, StateGlobal
from ecojax.spaces import Continuous, Discrete
from ecojax.utils import jprint, jprint_and_breakpoint


def get_eco_loop_fn(
    env: BaseEcoEnvironment,
    agent_species: BaseAgentSpecies,
    n_timesteps: int,
    do_render: bool = True,
) -> Callable[[StateGlobal], StateGlobal]:
    """
    Get the eco_loop function that will run the simulation of the environment and the agents.

    Args:
        env (BaseEcoEnvironment): the environment
        agent_species (BaseModel): the species of agents
        n_timesteps (int): the number of timesteps to run
        do_render (bool, optional): whether to render the environment. Defaults to True.

    Returns:
        Callable[[StateGlobal], StateGlobal]: the eco_loop function
    """

    

    def callback(global_state: StateGlobal) -> None:
        # Print the timestep
        print(f"timestep_run: {global_state.timestep_run}")
        
        # Render the environment
        if do_render:
            env.render(state=global_state.state_env)
    
    @jax.jit
    def step_global(global_state: StateGlobal) -> StateGlobal:
        key_random = global_state.key_random

        # Callback
        # jax.debug.callback(callback, global_state)
        jax.experimental.io_callback(callback, global_state, global_state)

        # Agents step
        key_random, subkey = random.split(key_random)
        new_state_species, actions = agent_species.react(
            state=global_state.state_species,
            batch_observations=global_state.observations,
            eco_information=global_state.eco_information,
            key_random=subkey,
        )

        # Env step
        key_random, subkey = random.split(key_random)
        new_state_env, new_observations, new_eco_information, new_done = env.step(
            state=global_state.state_env,
            actions=actions,
            key_random=subkey,
        )

        # Log the metrics
        pass

        # Return the new global state
        key_random, subkey = random.split(key_random)
        return StateGlobal(
            state_env=new_state_env,
            state_species=new_state_species,
            observations=new_observations,
            eco_information=new_eco_information,
            timestep_run=global_state.timestep_run + 1,
            done=new_done,
            key_random=subkey,
        )

    @jax.jit
    def do_stop_loop(global_state: StateGlobal) -> bool:
        return jax.numpy.logical_or(
            global_state.done,
            global_state.timestep_run >= n_timesteps,
        )
        
    def eco_loop(key_random: jnp.ndarray) -> None:

        # Initialize the environment
        print("Starting environment...")
        key_random, subkey = random.split(key_random)
        (
            state_env,
            observations,
            eco_information,
            done,
            info_env,
        ) = env.reset(key_random=subkey)

        # Initialize the species
        print("Starting agents...")
        key_random, subkey = random.split(key_random)
        state_species = agent_species.reset(key_random=subkey)

        # Initialize the global state
        global_state_initial = StateGlobal(
            state_env=state_env,
            state_species=state_species,
            observations=observations,
            eco_information=eco_information,
            timestep_run=0,
            done=done,
            key_random=subkey,
        )

        # Run the simulation
        # while not do_stop_loop(global_state_initial):
        #     global_state_initial = step_global(global_state_initial)
        while_loop(
            cond_fun=do_stop_loop,
            body_fun=step_global,
            init_val=global_state_initial,
        )
    return eco_loop
