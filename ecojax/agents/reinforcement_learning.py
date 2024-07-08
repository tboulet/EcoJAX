from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Dict, List, Tuple, Type

import jax
from jax import random, tree_map
import jax.numpy as jnp
import numpy as np
from flax import struct
import flax.linen as nn
import optax
from flax.training import train_state
from flax.struct import PyTreeNode, dataclass


from ecojax.agents.base_agent_species import AgentSpecies
from ecojax.core.eco_info import EcoInformation
from ecojax.models.base_model import BaseModel
from ecojax.evolution.mutator import mutate_scalar, mutation_gaussian_noise
from ecojax.types import ActionAgent, ObservationAgent
import ecojax.spaces as spaces
from ecojax.utils import jprint


@dataclass
class HyperParametersRL:
    # The learning rate of the agentnt
    lr: float
    # The discount factor of the agent
    gamma: float
    # The exploration rate of the agent
    epsilon: float
    # The mutation strength of the agent
    strength_mutation: float


@dataclass
class AgentRL(PyTreeNode):
    # The age of the agent, in number of timesteps
    age: int

    # The initial parameters of the neural networ
    params: Dict[str, jnp.ndarray]

    # The hyperparameters of the agent
    hp: HyperParametersRL
    
    # The last observation of the agent
    obs_last: jnp.ndarray
    
    # The last action of the agent
    action_last: int


@dataclass
class StateSpeciesRL:
    # The agents of the species
    agents: AgentRL  # Batched

    # The training state of the species
    tr_state: train_state.TrainState  # Batched


class RL_AgentSpecies(AgentSpecies):
    """A species of agents that learn with reinforcement learning."""

    def __init__(
        self,
        config: Dict,
        n_agents_max: int,
        n_agents_initial: int,
        observation_space_dict: Dict[str, spaces.EcojaxSpace],
        observation_class: Type[ObservationAgent],
        n_actions: int,
        model_class: Type[BaseModel],
        config_model: Dict,
    ):
        super().__init__(
            config=config,
            n_agents_max=n_agents_max,
            n_agents_initial=n_agents_initial,
            observation_space_dict=observation_space_dict,
            observation_class=observation_class,
            n_actions=n_actions,
            model_class=model_class,
            config_model=config_model,
        )
        self.model = model_class(
            observation_space_dict=observation_space_dict,
            observation_class=observation_class,
            n_actions=n_actions,
            return_modes=["q_values"],
            **config_model,
        )
        print(self.model.get_table_summary())
        self.innate : bool = self.config["innate"]
        self.name_exploration : str = self.config["name_exploration"]
        
    def reset(self, key_random: jnp.ndarray) -> StateSpeciesRL:
        # Initialize the state
        key_random, subkey = random.split(key_random)
        batch_keys = jax.random.split(subkey, self.n_agents_max)
        batch_agents = jax.vmap(self.init_agent)(batch_keys)
        batch_tr_states = jax.vmap(self.init_tr_state)(batch_agents)
        return StateSpeciesRL(
            agents=batch_agents,
            tr_state=batch_tr_states,
        )

    def init_hp(self) -> HyperParametersRL:
        """Get the initial hyperparameters of the agent from the config"""
        return HyperParametersRL(**self.config["hp_initial"])

    def init_agent(
        self,
        key_random: jnp.ndarray,
        params: Dict[str, jnp.ndarray] = None,
        hp: HyperParametersRL = None,
    ) -> AgentRL:
        """Create a new agent.

        Args:
            key_random (jnp.ndarray): the random key
            params (Dict[str, jnp.ndarray], optional): the NN parameters. Defaults to None (will be initialized randomly)
            hp (HyperParametersRL, optional): the hyperparameters of the agent. Defaults to None (will be initialized from the config).

        Returns:
            AgentRL: the new agent
        """
        if params is None:
            key_random, subkey = random.split(key_random)
            variables = self.model.get_initialized_variables(subkey)
            params = variables.get("params", {})
        if hp is None:
            hp = self.init_hp()
        key_random, subkey = random.split(key_random)
        return AgentRL(
            age=0,
            params=params,
            hp=hp,
            obs_last=self.model.sample_observation(subkey),
            action_last=0, # dummy action
        )

    def init_tr_state(self, agent: AgentRL) -> train_state.TrainState:
        """Initialize the training state of the agent."""
        tx = optax.adam(learning_rate=agent.hp.lr)
        tr_state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=agent.params, tx=tx
        )
        return tr_state

    def react(
        self,
        state: StateSpeciesRL,
        batch_observations: ObservationAgent,  # Batched
        eco_information: EcoInformation,
        key_random: jnp.ndarray,
    ) -> Tuple[
        StateSpeciesRL,
        ActionAgent,  # Batched
    ]:

        # Apply the mutation
        batch_keys = random.split(key_random, self.n_agents_max)
        batch_agents_mutated = jax.vmap(self.mutate_state_agent)(
            agent=state.agents, key_random=batch_keys
        )

        # Transfer the genes from the parents to the childs component by component using jax.tree_map
        are_newborns_agents = eco_information.are_newborns_agents
        indexes_parents = eco_information.indexes_parents  # (n_agents, n_parents)
        if indexes_parents.shape == (self.n_agents_max, 1):
            indexes_parents = indexes_parents.squeeze(axis=-1)  # (n_agents,)
        else:
            raise NotImplementedError(
                f"Invalid shape for indexes_parents: {indexes_parents.shape}"
            )

        def manage_genetic_component_inheritance(
            genes: jnp.ndarray,
            genes_mutated: jnp.ndarray,
        ):
            """The function that manages the inheritance of genetic components.
            It will be apply to each genetic component (JAX array of shape (n_agents, *gene_shape)) of the agents.
            It replace the genes by the mutated genes of the parents but only for the newborn agents.

            Args:
                genes (jnp.ndarray): a JAX array of shape (n_agents, *gene_shape)
                genes_mutated (jnp.ndarray): a JAX array of shape (n_agents, *gene_shape), which are mutated genes

            Returns:
                genes_inherited (jnp.ndarray): a JAX array of shape (n_agents, *gene_shape), which are the genes of the population after inheritance
            """
            mask = are_newborns_agents
            for _ in range(genes.ndim - 1):
                mask = jnp.expand_dims(mask, axis=-1)
            genes_inherited = jnp.where(
                mask,
                genes_mutated[indexes_parents],
                genes,
            )
            return genes_inherited

        new_batch_agents = tree_map(
            manage_genetic_component_inheritance,
            state.agents,
            batch_agents_mutated,
        )

        new_state_species = StateSpeciesRL(
            agents=new_batch_agents,
            tr_state=state.tr_state,
        )

        # Agent-wise reaction
        key_random, subkey = random.split(key_random)
        new_state_species, batch_actions = self.react_agents(
            key_random=subkey,
            state_species=new_state_species,
            batch_observations=batch_observations,
        )

        return new_state_species, batch_actions

    # =============== Mutation methods =================

    def mutate_state_agent(self, agent: AgentRL, key_random: jnp.ndarray) -> AgentRL:

        # Mutate the hyperparameters
        key_random, *subkeys = random.split(key_random, 5)
        hp = HyperParametersRL(
            lr=mutate_scalar(value=agent.hp.lr, range=(0, None), key_random=subkeys[0]),
            gamma=mutate_scalar(
                value=agent.hp.gamma, range=(0, 1), key_random=subkeys[1]
            ),
            epsilon=mutate_scalar(
                value=agent.hp.epsilon, range=(0, 0.5), key_random=subkeys[2]
            ),
            strength_mutation=mutate_scalar(
                value=agent.hp.strength_mutation, range=(0, None), key_random=subkeys[3]
            ),
        )

        # If "innate" learning is enable, inherit the parameters from the parents with a small mutation
        if self.innate:
            key_random, subkey = random.split(key_random)
            params = mutation_gaussian_noise(
                arr=agent.params,
                strength_mutation=agent.hp.strength_mutation,
                key_random=subkey,
            )
            return self.init_agent(
                key_random=key_random,
                params=params,
                hp=hp,
            )
        # Else, reset the parameters
        else:
            return self.init_agent(
                key_random=key_random,
                hp=hp,
            )

    # =============== Agent inference & learning methods =================

    def react_agents(
        self,
        key_random: jnp.ndarray,
        state_species: StateSpeciesRL,
        batch_observations: ObservationAgent,  # Batched
    ) -> jnp.ndarray:

        def react_single_agent(
            key_random: jnp.ndarray,
            agent: AgentRL,
            tr_state: train_state.TrainState,
            obs: jnp.ndarray,
        ) -> jnp.ndarray:
            # =============== Inference part =================
            # Compute Q values
            q_values, = self.model.apply(
                variables={"params": agent.params},
                obs=obs,
                key_random=key_random,
            )
            # Pick an action using the exploration method
            if self.name_exploration == "epsilon_greedy":
                key_random, subkey1, subkey2 = random.split(key_random, 3)
                action = jax.lax.cond(
                    random.uniform(subkey1) < agent.hp.epsilon,
                    lambda _: random.randint(subkey2, shape=(), minval=0, maxval=self.n_actions),
                    lambda _: jnp.argmax(q_values),
                    operand=None,
                )
            elif self.name_exploration == "greedy":
                action = jnp.argmax(q_values)
            elif self.name_exploration == "softmax":
                key_random, subkey = random.split(key_random)
                action = random.categorical(subkey, logits=q_values)
            else:
                raise NotImplementedError(
                    f"Exploration method {self.name_exploration} not implemented"
                )
            # ============== Learning part =================
            # Compute the reward
            reward_last = 0.1 # TODO: Implement the reward
            # Perform the learning step
            def loss_fn(params):
                q_values_last, = self.model.apply(
                    variables={"params": params},
                    obs=agent.obs_last,
                    key_random=key_random,
                )
                q_value_last = q_values_last[agent.action_last]
                target = reward_last + agent.hp.gamma * jnp.max(q_values)
                loss = (q_value_last - target) ** 2
                return loss
            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(tr_state.params)
            tr_state = tr_state.apply_gradients(grads=grads)
            # Update the agent's state
            agent.replace(
                params=tr_state.params,
                age=agent.age + 1,
                obs_last=obs,
                action_last=action,
                )
            pass  # TODO: Implement the learning part
            # Update the agent's state and act
            return agent, tr_state, action

        batch_keys = random.split(key_random, self.n_agents_max)
        batch_agents, batch_tr_state, batch_actions = jax.vmap(react_single_agent)(
            key_random=batch_keys,
            agent=state_species.agents,
            tr_state=state_species.tr_state,
            obs=batch_observations,
        )

        new_state_species = StateSpeciesRL(
            agents=batch_agents,
            tr_state=batch_tr_state,
        )
        return new_state_species, batch_actions
