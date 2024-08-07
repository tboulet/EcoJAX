from abc import ABC, abstractmethod
from functools import partial
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import jax
from jax import random, tree_map
import jax.numpy as jnp
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from flax import struct
import flax.linen as nn
import optax
from flax.training import train_state
from flax.struct import PyTreeNode, dataclass


from ecojax.agents.base_agent_species import AgentSpecies
from ecojax.core.eco_info import EcoInformation
from ecojax.metrics.aggregators import Aggregator
from ecojax.models.base_model import BaseModel
from ecojax.evolution.mutator import mutate_scalar, mutation_gaussian_noise
from ecojax.models.mlp import MLP_Model
from ecojax.types import ActionAgent, ObservationAgent
import ecojax.spaces as spaces
from ecojax.utils import instantiate_class, jprint


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
    # The hyperparameters of the agent
    hp: HyperParametersRL
    # The initial parameters of the neural network
    params_sensor: Dict[str, jnp.ndarray]
    # The parameters of the decision model
    params_decision: Dict[str, jnp.ndarray]
    # In case of initial weights transmission, the initial parameters of the decision model
    params_decision_initial: Optional[Dict[str, jnp.ndarray]]
    # The parameters of the reward model (not necessarily a NN, can be a more simple model)
    params_reward: Dict[str, jnp.ndarray]
    # Whether the agent is existing
    do_exist: bool
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

    # The lifespan and population aggregators
    metrics_lifespan: List[PyTreeNode]
    metrics_population: List[PyTreeNode]


class RewardModel(BaseModel):
    """The reward model of the agent. It maps the observation to the reward."""

    space_input: spaces.DictSpace
    space_output: spaces.ContinuousSpace
    func_weight_diff: str

    def obs_to_encoding(
        self,
        x: Dict[str, jnp.ndarray],
        key_random: jnp.ndarray,
    ) -> Tuple[jnp.ndarray]:
        """The encoding of the reward model. It receives as input a dictionnary containing the following :

        - "obs" : the observation of the agent
        - "obs_next" : the next observation of the agent

        It will then extract the scalar components of the observations and compute the reward from them only.

        Args:
            x (Dict[str, jnp.ndarray]): a dictionnary containing the observations of the agent

        Returns:
            jnp.ndarray: the reward of the agent
        """
        assert isinstance(
            self.space_input, spaces.DictSpace
        ), f"Expected a TupleSpace, got {self.space_input}"
        space_obs: spaces.DictSpace = self.space_input.dict_space["obs"]
        space_next_obs: spaces.DictSpace = self.space_input.dict_space["obs_next"]
        # assert space_obs == space_next_obs, "The observation spaces must be the same"

        reward = jnp.zeros((1,))
        for name_space, space in space_obs.dict_space.items():
            if (
                isinstance(space, spaces.ContinuousSpace) and space.shape == ()
            ):  # Scalar space
                obs_component = x["obs"][name_space]
                obs_component_next = x["obs_next"][name_space]
                diff_obs = obs_component_next - obs_component
                alpha_diff = self.param(
                    name=f"alpha_diff_{name_space}",
                    init_fn=nn.initializers.ones,
                    shape=(),
                    dtype=jnp.float32,
                )
                if self.func_weight_diff == "constant":
                    reward += alpha_diff * diff_obs
                elif self.func_weight_diff == "linear":
                    beta_diff = self.param(
                        name=f"beta_diff_{name_space}",
                        init_fn=nn.initializers.ones,
                        shape=(),
                        dtype=jnp.float32,
                    )
                    reward += (alpha_diff + obs_component * beta_diff) * diff_obs
                else:
                    raise NotImplementedError(
                        f"Function {self.func_weight_diff} not implemented"
                    )
        return reward


class RL_AgentSpecies(AgentSpecies):
    """A species of agents that learn with reinforcement learning."""

    def __init__(
        self,
        config: Dict,
        n_agents_max: int,
        n_agents_initial: int,
        observation_space: spaces.EcojaxSpace,
        action_space: spaces.DiscreteSpace,
        model_class: Type[BaseModel],
        config_model: Dict,
    ):
        super().__init__(config=config,
            n_agents_max=n_agents_max,
            n_agents_initial=n_agents_initial,
            observation_space=observation_space,
            action_space=action_space,
            model_class=model_class,
            config_model=config_model,
        )
        assert isinstance(action_space, spaces.DiscreteSpace), f"Only DiscreteSpace is supported for now, got {action_space}"
        self.n_actions = action_space.n

        # Sensor model : this model converts the observation to "sensations", which are some internal neuro-evolved representation of the observation
        n_sensations = config["n_sensations"]
        self.sensor_model = model_class(
            space_input=observation_space,
            space_output=spaces.ContinuousSpace(
                n_sensations,
            ),
            **config_model,
        )
        print(f"Sensor model: {self.sensor_model.get_table_summary()}")

        # Decision model : this RL model converts the sensations to the decision (actions)
        config_decision_model = config["decision_model"]
        self.decision_model = MLP_Model(
            space_input=spaces.ContinuousSpace(n_sensations),
            space_output=spaces.ContinuousSpace(self.n_actions),
            **config_decision_model,
        )
        print(f"Decision model: {self.decision_model.get_table_summary()}")

        # Reward model : this neuro-evolved model converts the observation variation to a reward used by the RL process
        self.reward_model = RewardModel(
            space_input=spaces.DictSpace(
                dict_space={
                    "obs": observation_space,
                    "obs_next": observation_space,
                }
            ),
            space_output=spaces.ContinuousSpace(shape=()),
            **config["reward_model"],
        )
        print(f"Reward model: {self.reward_model.get_table_summary()}")

        # Hyperparameters
        self.mode_weights_transmission: str = self.config["mode_weights_transmission"]
        self.name_exploration: str = self.config["name_exploration"]

        # Metrics parameters
        self.names_measures: List[str] = sum(
            [names for type_measure, names in config["metrics"]["measures"].items()], []
        )

    def reset(self, key_random: jnp.ndarray) -> StateSpeciesRL:

        # Initialize the metrics
        self.aggregators_lifespan: List[Aggregator] = []
        list_metrics_lifespan: List[PyTreeNode] = []
        for config_agg in self.config["metrics"]["aggregators_lifespan"]:
            agg: Aggregator = instantiate_class(**config_agg)
            self.aggregators_lifespan.append(agg)
            list_metrics_lifespan.append(agg.get_initial_metrics())

        self.aggregators_population: List[Aggregator] = []
        list_metrics_population: List[PyTreeNode] = []
        for config_agg in self.config["metrics"]["aggregators_population"]:
            agg: Aggregator = instantiate_class(**config_agg)
            self.aggregators_population.append(agg)
            list_metrics_population.append(agg.get_initial_metrics())

        # Initialize the state
        key_random, subkey = random.split(key_random)
        batch_keys = jax.random.split(subkey, self.n_agents_max)
        batch_agents = jax.vmap(self.init_agent)(
            key_random=batch_keys,
            do_exist=jnp.arange(self.n_agents_max)
            < self.n_agents_initial,  # by convention, the first n_agents_initial agents exist
        )
        batch_tr_states = jax.vmap(self.init_tr_state)(batch_agents)
        return StateSpeciesRL(
            agents=batch_agents,
            tr_state=batch_tr_states,
            metrics_lifespan=list_metrics_lifespan,
            metrics_population=list_metrics_population,
        )

    def init_hp(self) -> HyperParametersRL:
        """Get the initial hyperparameters of the agent from the config"""
        return HyperParametersRL(**self.config["hp_initial"])

    def init_agent(
        self,
        key_random: jnp.ndarray,
        do_exist: bool = True,
        hp: Optional[HyperParametersRL] = None,
        params_sensor: Optional[Dict[str, jnp.ndarray]] = None,
        params_decision: Optional[Dict[str, jnp.ndarray]] = None,
        params_decision_initial: Optional[Dict[str, jnp.ndarray]] = None,
        params_reward: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> AgentRL:
        """Create a new agent.

        Args:
            key_random (jnp.ndarray): the random key
            do_exist (bool, optional): whether the agent actually exists in the simulation. Defaults to True.
            hp (HyperParametersRL, optional): the hyperparameters of the agent. Defaults to None (will be initialized from the config).
            params_sensor (Dict[str, jnp.ndarray], optional): the sensor NN parameters. Defaults to None (will be initialized randomly)
            params_decision (Dict[str, jnp.ndarray], optional): the decision NN parameters. Defaults to None (will be initialized randomly)
            params_decision_initial (Dict[str, jnp.ndarray], optional): the initial decision NN parameters. Defaults to None (no initial weights transmission)
            params_reward (Dict[str, jnp.ndarray], optional): the parameters of the reward model. Defaults to None (will be initialized randomly).

        Returns:
            AgentRL: the new agent
        """
        if hp is None:
            hp = self.init_hp()
        if params_sensor is None:
            key_random, subkey = random.split(key_random)
            variables = self.sensor_model.get_initialized_variables(subkey)
            params_sensor = variables.get("params", {})
        if params_decision is None:
            key_random, subkey = random.split(key_random)
            variables = self.decision_model.get_initialized_variables(subkey)
            params_decision = variables.get("params", {})
        if params_decision_initial is None and self.mode_weights_transmission == "initial":
            key_random, subkey = random.split(key_random)
            variables = self.decision_model.get_initialized_variables(subkey)
            params_decision_initial = variables.get("params", {})
        if params_reward is None:
            key_random, subkey = random.split(key_random)
            variables = self.reward_model.get_initialized_variables(subkey)
            params_reward = variables.get("params", {})
        key_random, subkey = random.split(key_random)
        obs_dummy = self.observation_space.sample(subkey)
        # internal_obs_dummy = self.model_sensor.apply(
        #     variables={"params": params_sensor},
        #     x=obs_dummy,
        #     key_random=subkey,
        # )
        return AgentRL(
            age=0,
            hp=hp,
            params_sensor=params_sensor,
            params_decision=params_decision,
            params_decision_initial=params_decision_initial,
            params_reward=params_reward,
            do_exist=do_exist,
            obs_last=obs_dummy,  # dummy observation
            action_last=-1,  # dummy action
        )

    def init_tr_state(self, agent: AgentRL) -> train_state.TrainState:
        """Initialize the training state of the agent."""
        tx = optax.sgd(
            learning_rate=1
        )  # set tx's learning rate as 1 and scale loss for pseudo-learning rate
        tr_state = train_state.TrainState.create(
            apply_fn=self.sensor_model.apply, params=agent.params_decision, tx=tx
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

        # Initialize the measures dictionnary. This will be used to store the measures of the environment at this step.
        dict_measures_all: Dict[str, jnp.ndarray] = {}

        # Apply the mutation
        batch_keys = random.split(key_random, self.n_agents_max)
        agents_mutated = jax.vmap(self.mutate_state_agent)(
            agent=state.agents, key_random=batch_keys
        )

        # Transfer the mutated genes from the parents to the childs component by component using jax.tree_map
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

        new_agents = tree_map(
            manage_genetic_component_inheritance,
            state.agents,
            agents_mutated,
        )

        # Update the do_exist attribute
        new_do_exist = (
            new_agents.do_exist & ~eco_information.are_just_dead_agents
        )  # Remove the agents that have just died
        new_do_exist = new_do_exist | are_newborns_agents  # Add the newborn agents
        new_agents = new_agents.replace(do_exist=new_do_exist)

        new_state = state.replace(agents=new_agents)

        # ================== Agent-wise reaction ==================
        key_random, subkey = random.split(key_random)
        new_state, batch_actions, dict_measures = self.react_agents(
            key_random=subkey,
            state=new_state,
            batch_observations=batch_observations,
        )
        dict_measures_all.update(dict_measures)

        # ============ Compute the metrics ============

        # Compute some measures
        key_random, subkey = random.split(key_random)
        dict_measures = self.compute_measures(
            state=state, state_new=new_state, key_random=subkey
        )
        dict_measures_all.update(dict_measures)

        # Set the measures to NaN for the agents that are not existing
        for name_measure, measures in dict_measures_all.items():
            if name_measure not in self.config["metrics"]["measures"]["global"]:
                dict_measures_all[name_measure] = jnp.where(
                    new_state.agents.do_exist,
                    measures,
                    jnp.nan,
                )

        # Update and compute the metrics
        new_state, dict_metrics = self.compute_metrics(
            state=state, state_new=new_state, dict_measures=dict_measures_all
        )
        info = {"metrics": dict_metrics}

        return new_state, batch_actions, info

    # =============== Mutation methods =================

    def mutate_state_agent(self, agent: AgentRL, key_random: jnp.ndarray) -> AgentRL:

        # Mutate the hyperparameters
        key_random, *subkeys = random.split(key_random, 5)
        new_hp = HyperParametersRL(
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

        # Mutate the sensor model
        key_random, subkey = random.split(key_random)
        new_params_sensor = mutation_gaussian_noise(
            arr=agent.params_sensor,
            strength_mutation=agent.hp.strength_mutation,
            key_random=subkey,
        )

        # Mutate the reward model
        key_random, subkey = random.split(key_random)
        new_params_reward = mutation_gaussian_noise(
            arr=agent.params_reward,
            strength_mutation=agent.hp.strength_mutation,
            key_random=subkey,
        )

        # Transmit (or not) the weights according to the mode
        if self.mode_weights_transmission == "initial":
            # Mute and transmit the initial weights
            new_params_decision_initial = mutation_gaussian_noise(
                arr=agent.params_decision_initial,
                strength_mutation=agent.hp.strength_mutation,
                key_random=key_random,
            )
            new_params_decision = new_params_decision_initial.copy()
        elif self.mode_weights_transmission == "final":
            # Transmit the final weights
            new_params_decision = agent.params_decision
            new_params_decision_initial = None
        elif self.mode_weights_transmission == "none":
            # Don't transmit any weights
            new_params_decision = None
            new_params_decision_initial = None
        else:
            raise NotImplementedError(
                f"Mode {self.mode_weights_transmission} not implemented"
            )

        return self.init_agent(
            key_random=key_random,
            hp=new_hp,
            params_sensor=new_params_sensor,
            params_decision=new_params_decision,
            params_decision_initial=new_params_decision_initial,
            params_reward=new_params_reward,
        )

    # =============== Agent inference & learning methods =================

    def react_agents(
        self,
        key_random: jnp.ndarray,
        state: StateSpeciesRL,
        batch_observations: ObservationAgent,  # Batched
    ) -> Tuple[StateSpeciesRL, ActionAgent]:  # Batched

        def react_single_agent(
            key_random: jnp.ndarray,
            agent: AgentRL,
            tr_state: train_state.TrainState,
            obs: jnp.ndarray,
        ) -> Tuple[
            AgentRL,
            train_state.TrainState,
            int,
            Dict[str, jnp.ndarray],
        ]:
            # =============== Inference part =================
            # Compute internal observation (sensation)
            internal_obs = self.sensor_model.apply(
                variables={"params": agent.params_sensor},
                x=obs,
                key_random=key_random,
            )
            internal_obs_last = self.sensor_model.apply(
                variables={"params": agent.params_sensor},
                x=agent.obs_last,
                key_random=key_random,
            )
            # Compute the Q-values
            q_values = self.decision_model.apply(
                variables={"params": agent.params_decision},
                x=internal_obs,
                key_random=key_random,
            )
            # Pick an action using the exploration method
            if self.name_exploration == "epsilon_greedy":
                key_random, subkey1, subkey2 = random.split(key_random, 3)
                action = jax.lax.cond(
                    random.uniform(subkey1) < agent.hp.epsilon,
                    lambda _: random.randint(
                        subkey2, shape=(), minval=0, maxval=self.n_actions
                    ),
                    lambda _: jnp.argmax(q_values),
                    operand=None,
                )
            elif self.name_exploration == "greedy":
                action = jnp.argmax(q_values)
            elif self.name_exploration == "softmax":
                key_random, subkey = random.split(key_random)
                temperature = self.config.get("temperature_softmax", 1.0)
                action = random.categorical(subkey, logits=q_values / temperature)
            else:
                raise NotImplementedError(
                    f"Exploration method {self.name_exploration} not implemented"
                )
            # ============== Learning part =================
            # Compute the reward
            key_random, subkey = random.split(key_random)
            reward_last = self.reward_model.apply(
                variables={"params": agent.params_reward},
                x={"obs": agent.obs_last, "obs_next": obs},
                key_random=subkey,
            )

            # Perform the RL step (1-step memory DQN)
            def loss_fn(params_decision):
                q_values_last = self.decision_model.apply(
                    variables={"params": params_decision},
                    x=internal_obs_last,
                    key_random=key_random,
                )
                q_value_last = q_values_last[agent.action_last]
                target = reward_last + agent.hp.gamma * jnp.max(q_values)
                loss_q = (q_value_last - target) ** 2
                loss_q *= (
                    agent.hp.lr
                )  # Scale the loss by the learning rate to simulate learning rate
                loss_q *= (
                    agent.age >= 1
                )  # Only learn at age 1 (and more) when you have already seen a transition
                return loss_q

            grad_fn = jax.value_and_grad(loss_fn)
            loss_q, grads = grad_fn(tr_state.params)
            tr_state = tr_state.apply_gradients(grads=grads)

            # Update the agent's state
            agent = agent.replace(
                params_decision=tr_state.params,
                age=agent.age + 1,
                obs_last=obs,
                action_last=action,
            )

            # ============== Measures ==============
            dict_measures = {
                "loss_q": loss_q,
                "q_values_max": jnp.max(q_values),
                "q_values_mean": jnp.mean(q_values),
                "q_values_median": jnp.median(q_values),
                "q_values_min": jnp.min(q_values),
                "target": reward_last + agent.hp.gamma * jnp.max(q_values),
                "reward": reward_last,
            }
            # Update the agent's state and act
            return agent, tr_state, action, dict_measures

        batch_keys = random.split(key_random, self.n_agents_max)
        new_agents, batch_tr_state, actions, dict_measures = jax.vmap(
            react_single_agent
        )(
            key_random=batch_keys,
            agent=state.agents,
            tr_state=state.tr_state,
            obs=batch_observations,
        )
        dict_measures.update(**new_agents.params_reward)

        new_state = state.replace(
            agents=new_agents,
            tr_state=batch_tr_state,
        )
        return new_state, actions, dict_measures

    # =============== Metrics methods =================

    def render(self, state: StateSpeciesRL, force_render: bool = False) -> None:
        """Do the rendering of the species. This can be a visual rendering or a logging of the state of any kind.

        Args:
            state (StateSpecies): the state of the species to render
            force_render (bool): whether to force the rendering even if the species is not in a state where it should be rendered
        """        
        # Log heatmaps of the weights
        try:
            weights = state.agents.params_decision["Dense_0"]["kernel"].mean(axis=0)
            n_obs, n_actions = weights.shape
            bias = state.agents.params_decision["Dense_0"]["bias"].mean(axis=0)
            plt.figure(figsize=(10, 8))
            sns.heatmap(weights, annot=True, cmap='viridis', cbar=True)
            plt.xlabel('Actions')
            plt.ylabel('Observations')
            plt.title('Heatmap of Weights')
            os.makedirs("logs/heatmaps", exist_ok=True)
            plt.savefig(f"logs/heatmaps/heatmap.png")

            x = np.arange(n_actions)  # the label locations
            width = 0.35  # the width of the bars
            sum_weights = np.sum(weights, axis=0)

            fig, ax = plt.subplots(figsize=(10, 6))
            bars1 = ax.bar(x - width/2, bias, width, label='Bias', color='skyblue')
            bars2 = ax.bar(x + width/2, sum_weights, width, label='Sum of Weights', color='lightgreen')

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_xlabel('Actions')
            ax.set_ylabel('Values')
            ax.set_title('Bias and Sum of Weights for Each Action')
            ax.set_xticks(x)
            ax.set_xticklabels(x)
            ax.legend()

            # Path to save the combined bar chart
            combined_bar_chart_path = os.path.join("logs/heatmaps/bias and sum weights.png")

            # Save the combined bar chart
            plt.savefig(combined_bar_chart_path)
            plt.close()

        except Exception as e:
            print(f"Error in agents render: {e}")
            
            
    def compute_measures(
        self,
        state: StateSpeciesRL,
        state_new: StateSpeciesRL,
        key_random: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray]:
        """Compute the measures of the agents."""
        dict_measures = {}
        for name_measure in self.names_measures:
            # Global measures
            pass
            # Immediate measures
            pass
            # State measures
            if "hp" in name_measure:
                strength_mutation = getattr(state.agents.hp, "strength_mutation")
                dict_measures["strength_mutation"] = strength_mutation
                dict_measures["log10/strength_mutation"] = jnp.log10(strength_mutation)
                dict_measures["lr"] = getattr(state.agents.hp, "lr")
                dict_measures["log10/lr"] = jnp.log10(getattr(state.agents.hp, "lr"))
                dict_measures["gamma"] = getattr(state.agents.hp, "gamma")
                dict_measures["epsilon"] = getattr(state.agents.hp, "epsilon")
            # Behavior measures
            pass

        return dict_measures
