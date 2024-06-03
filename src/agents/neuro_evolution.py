from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, List, Tuple, Type

import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from flax import struct
import flax.linen as nn


from src.agents import BaseAgentSpecies
from src.models.base_model import BaseModel
from src.types_base import ActionAgent, ObservationAgent, StateAgent
import src.spaces as spaces


@struct.dataclass
class StateAgentEvolutionary(StateAgent):
    # The age of the agent, in number of timesteps
    age: int

    # The parameters of the neural network corresponding to the agent
    params: Dict[str, jnp.ndarray]


def update_states(
    batch_state_agents: StateAgent,  # Batched
    list_idx_agents: List[int],
    list_new_states: List[StateAgent],
) -> StateAgent:  # Batched
    """Update the state of a batch of agents with new values.

    Args:
        batch_state_agents (StateAgent): the state of the agents
        list_idx_agents (List[int]): the list of indices of the agents to update
        list_new_values (List[StateAgent]): the list of new values for the agents

    Returns:
        StateAgent: the updated state of the agents
    """
    for idx_agent, new_state in zip(list_idx_agents, list_new_states):
        state = batch_state_agents[idx_agent]
        for key in new_state.keys():
            state = state.set(key, new_state[key])
        batch_state_agents = batch_state_agents.at[idx_agent].set(state)
    return batch_state_agents


class NeuroEvolutionAgentSpecies(BaseAgentSpecies):
    """A species of agents that evolve their neural network weights."""

    def init(self, key_random: jnp.ndarray) -> None:

        # Initialize the state
        def init_single_agent(
            model: BaseModel,
            key_random: jnp.ndarray,
        ) -> jnp.ndarray:
            variables = model.get_initialized_variables(key_random)
            params = variables.get("params", {})
            return StateAgentEvolutionary(age=0, params=params)

        key_random, subkey = random.split(key_random)
        batch_keys = jax.random.split(subkey, self.n_agents_max)
        init_many_agents = jax.vmap(init_single_agent, in_axes=(None, 0))
        self.batch_state_agents = init_many_agents(
            self.model,
            batch_keys,
        )

    def react(
        self,
        key_random: jnp.ndarray,
        batch_observations: ObservationAgent,  # Batched
        dict_reproduction: Dict[int, List[int]],
    ) -> jnp.ndarray:

        # Reproduction part
        # for idx_agent_newborn, list_idx_parents in dict_reproduction.items():
        #     if len(list_idx_parents) == 0:
        #         pass
        #     elif len(list_idx_parents) == 1:
        #         agent_state = self.create_mutated_agent(
        #             idx_parent=list_idx_parents[0],
        #             key_random=key_random,
        #         )
                
        #         # Update the agent corresponding to the newborn with the new state
        #         self.batch_state_agents = update_states(
        #             batch_state_agents=self.batch_state_agents,
        #             list_idx_agents=[idx_agent_newborn],
        #             list_new_states=[agent_state],
        #         )

        #     elif len(list_idx_parents) == 2:
        #         agent_state = self.create_crossover_agent(
        #             idx_parent1=list_idx_parents[0],
        #             idx_parent2=list_idx_parents[1],
        #             key_random=key_random,
        #         )

        #     else:
        #         raise ValueError(
        #             f"list_idx_parents has length {len(list_idx_parents)}. Not supported."
        #         )
                
            

        # Agent-wise acting part
        batch_keys = random.split(key_random, self.n_agents_max)
        self.batch_state_agents, batch_actions = self.react_agents(
            batch_keys=batch_keys,
            batch_observations=batch_observations,
            batch_state_agents=self.batch_state_agents,
        )

        return batch_actions

    # =============== Reproduction methods =================

    def create_mutated_agent(
        self,
        idx_parent: int,
        key_random: jnp.ndarray,
    ) -> StateAgentEvolutionary:
        # Copy the parent's params and mutate them
        state_agent_parent = self.batch_state_agents[idx_parent]
        params = state_agent_parent.params
        return StateAgentEvolutionary(
            age=0,
            params=self.mutate_params(
                params=state_agent_parent.weights,
                key_random=key_random,
            ),
        )

    def mutate_params(
        self,
        params: Dict[str, jnp.ndarray],
        key_random: jnp.ndarray,
        factor_mutation: float = 0.01,
    ) -> Dict[str, jnp.ndarray]:
        """Mutate the params of an NN model by adding a small amount of noise.

        Args:
            params (Dict[str, jnp.ndarray]): the params of the model
            key_random (jnp.ndarray): the random key used for generating the noise
            factor_mutation (float): the factor by which to scale the noise

        Returns:
            Dict[str, jnp.ndarray]: the mutated params
        """
        params_mutated = {}
        for key, weight in params.items():
            key_random, subkey = random.split(key_random)
            noise = jax.random.normal(subkey, weight.shape) * factor_mutation
            params_mutated[key] = weight + noise 
        return params_mutated

    # =============== Agent creation methods =================

    @partial(jax.jit, static_argnums=(0,))
    def react_agents(
        self,
        batch_keys: jnp.ndarray,
        batch_observations: ObservationAgent,  # Batched
        batch_state_agents: StateAgentEvolutionary,  # Batched
    ) -> jnp.ndarray:

        def react_single_agent(
            key_random: jnp.ndarray,
            obs: jnp.ndarray,
            state_agent: StateAgentEvolutionary,
        ) -> jnp.ndarray:
            # Inference part
            action, prob_action = self.model.apply(
                variables={"params": state_agent.params},
                obs=obs,
                key_random=key_random,
            )
            # Learning part
            pass
            # Update the agent's state and act
            return state_agent, action

        react_many_agents = jax.vmap(react_single_agent, in_axes=(0, 0, 0))
        batch_state_agents, batch_actions = react_many_agents(
            batch_keys,
            batch_observations,
            batch_state_agents,
        )

        return batch_state_agents, batch_actions
