from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import jax
from jax import random, tree_map, tree_structure
import jax.numpy as jnp
from jax.numpy import ndarray
from matplotlib import pyplot as plt, table
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
from ecojax.models.base_model import BaseModel, FlattenAndConcatModel
from ecojax.evolution.mutator import mutate_scalar, mutation_gaussian_noise
from ecojax.models.mlp import MLP_Model
from ecojax.agents.model_reward import RewardModel
from ecojax.types import ActionAgent, ObservationAgent
import ecojax.spaces as spaces
from ecojax.utils import instantiate_class, jbreakpoint, jprint


@dataclass
class HyperParametersAdaRL:
    # The mutation strength of the agent
    strength_mutation: float


@dataclass
class AgentAdaRL(PyTreeNode):
    # The age of the agent, in number of timesteps
    age: int
    # The parameters of the neural network corresponding to the agent
    params: Dict[str, jnp.ndarray]
    # The hyperparameters of the agent
    hp: HyperParametersAdaRL
    # The fruit value table of the agent, of shape (4,)
    table_value_fruits: jnp.ndarray
    # The initial fruit value table of the agent, of shape (4,)
    table_value_fruits_initial: jnp.ndarray
    # The parameters of the reward model (not necessarily a NN, can be a more simple model)
    params_reward: Dict[str, jnp.ndarray]
    # Whether the agent is existing
    do_exist: bool
    # The last observation of the agent
    obs_last: jnp.ndarray
    # The last action of the agent
    action_last: int


@dataclass
class StateSpeciesAdaRL:
    # The agents of the species
    agents: AgentAdaRL  # Batched

    # The lifespan and population aggregators
    metrics_lifespan: List[PyTreeNode]
    metrics_population: List[PyTreeNode]


class AdaptiveRL_AgentSpecies(AgentSpecies):
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
        super().__init__(
            config=config,
            n_agents_max=n_agents_max,
            n_agents_initial=n_agents_initial,
            observation_space=observation_space,
            action_space=action_space,
            model_class=model_class,
            config_model=config_model,
        )
        assert isinstance(
            action_space, spaces.DiscreteSpace
        ), f"Only DiscreteSpace is supported for now, got {action_space}"
        self.n_actions = action_space.n

        # Hyperparameters
        self.mode_weights_transmission: str = self.config["mode_weights_transmission"]
        self.n_latent_fruit: int = self.config["n_latent_fruit"]

        self.do_include_fruit: bool = self.config["do_include_fruit"]
        self.do_include_value_global: bool = self.config["do_include_value_global"]
        self.do_include_value_fruit: bool = self.config["do_include_value_fruit"]
        self.do_include_novelty_hunger: bool = self.config["do_include_novelty_hunger"]
        self.list_channels_visual_field: List[str] = self.config[
            "list_channels_visual_field"
        ]
        assert all(
            [f"fruits_{i}" in self.list_channels_visual_field for i in range(4)]
        ), "The visual field must contain the fruits channels"

        # Fruit model : this model will convert the observations related to a fruit to a value
        assert isinstance(
            self.observation_space, spaces.DictSpace
        ), "The observation space must be a DictSpace"
        assert (
            "visual_field" in self.observation_space.dict_space
        ), "The observation space must contain a 'visual_field' component"
        S, S_, _ = self.observation_space.dict_space["visual_field"].shape
        assert S == S_, "The visual field must be square"

        if "novelty_hunger" not in self.observation_space.dict_space:
            self.do_include_novelty_hunger = False
        self.C_input_visual_field = len(self.list_channels_visual_field)
        self.C_input_fruit_model = (
            self.C_input_visual_field
            - 4
            + int(self.do_include_fruit)
            + int(self.do_include_value_global)
            + int(self.do_include_value_fruit)
        )  # Get the number of channels of the fruit model : C_input_visual_field - 4 (remove the fruits) + sum_include_x (add the observations)

        self.space_observation_individual_fruit = deepcopy(self.observation_space)
        if "novelty_hunger" in self.space_observation_individual_fruit.dict_space:
            self.space_observation_individual_fruit.dict_space[
                "novelty_hunger"
            ].shape = ()  # Only fruit i will be included in the novelty_hunger of fruit i
        self.space_observation_individual_fruit.dict_space["visual_field"].shape = (
            S,
            S,
            self.C_input_fruit_model,
        )

        

        # Decision model : this neuro-evolved model converts the observation to an action
        # TODO : implement the decision model

        # Model : this model average the fruit output to obtain the logits
        class ModelFruitAverager(BaseModel):
            """The model of the agent, that will average the fruit values to obtain the logits of the actions.

            Args:
                space_input (spaces.DictSpace): the input space of the model
                space_output (spaces.ContinuousSpace): the output space of the model
            """

            space_input: spaces.DictSpace
            space_output: spaces.ContinuousSpace

            def obs_to_encoding(
                self_averager: "ModelFruitAverager",
                x: Dict[str, jnp.ndarray],
                key_random: jnp.ndarray,
            ) -> Tuple[jnp.ndarray]:

                # Create fruit model
                model_fruit = model_class(
                    space_input=self.space_observation_individual_fruit,
                    space_output=spaces.ContinuousSpace(
                        shape=(self.n_actions,)
                    ),  # TODO : change this to arbitrary (k,) shape
                    **config_model,
                )
                        
                # Compute the values of the fruits
                id_fruit_to_map_fruit_i_value = {}
                map_global_value = jnp.zeros((S, S))
                table_value_fruits = x["table_value_fruits"]
                for id_fruit in range(4):
                    idx_channel_fruit_i = self.list_channels_visual_field.index(
                        f"fruits_{id_fruit}"
                    )
                    map_fruit_i_value = (
                        table_value_fruits[id_fruit]
                        * x["visual_field"][:, :, idx_channel_fruit_i]
                    )
                    id_fruit_to_map_fruit_i_value[id_fruit] = map_fruit_i_value
                    map_global_value += map_fruit_i_value

                # Create the list of fruit-obs
                list_x_fruit = []
                for id_fruit in range(4):

                    # Compute fruit-obs for each fruit
                    x_fruit: Dict[str, jnp.ndarray] = {}
                    # Compute fruit-visual_field for each fruit
                    visual_field = x["visual_field"]
                    visual_field_fruit_i = jnp.zeros((S, S, self.C_input_fruit_model))
                    j = 0
                    for i, name_channel in enumerate(self.list_channels_visual_field):
                        # Add the non-fruit related channels
                        if not name_channel.startswith("fruits_"):
                            visual_field_fruit_i.at[:, :, j].set(visual_field[:, :, i])
                            j += 1
                        # Add the fruit presence channel
                        if name_channel == f"fruits_{id_fruit}":
                            if self.do_include_fruit:
                                visual_field_fruit_i.at[:, :, j].set(
                                    visual_field[:, :, i]
                                )
                                j += 1
                    # Add global value channel
                    if self.do_include_value_global:
                        visual_field_fruit_i.at[:, :, j].set(map_global_value)
                        j += 1
                    # Add fruit value channel
                    if self.do_include_value_fruit:
                        visual_field_fruit_i.at[:, :, j].set(
                            id_fruit_to_map_fruit_i_value[id_fruit]
                        )
                        j += 1
                    x_fruit["visual_field"] = visual_field_fruit_i

                    # Add novelty hunger scalar
                    if self.do_include_novelty_hunger:
                        x_fruit["novelty_hunger"] = x["novelty_hunger"][id_fruit]

                    # Add other scalars
                    for name_obs, component_obs in x.items():
                        if not name_obs in [
                            "visual_field",
                            "novelty_hunger",
                            "table_value_fruits",
                        ]:
                            x_fruit[name_obs] = component_obs

                    # Add the fruit to the list
                    list_x_fruit.append(x_fruit)

                # Apply the fruit model to each fruit
                list_encoding_fruit = []
                for x_fruit in list_x_fruit:
                    encoding_fruit = model_fruit(
                        x=x_fruit, key_random=key_random
                    )
                    list_encoding_fruit.append(encoding_fruit)
                
                # Average the fruit encodings
                encoding = (
                    list_encoding_fruit[0]
                    + list_encoding_fruit[1]
                    + list_encoding_fruit[2]
                    + list_encoding_fruit[3]
                ) / 4
                return encoding

        self.space_observation_fruit_averager = deepcopy(self.observation_space)
        self.space_observation_fruit_averager.dict_space["table_value_fruits"] = (
            spaces.ContinuousSpace(shape=(4,))
        )
        self.model = ModelFruitAverager(
            space_input=self.space_observation_fruit_averager,
            space_output=spaces.ContinuousSpace(shape=(self.n_actions,)),
            # TODO : add decision model and inbetween here
        )
        print(f"Model: {self.model.get_table_summary()}")

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

        # Metrics parameters
        self.names_measures: List[str] = sum(
            [names for type_measure, names in config["metrics"]["measures"].items()], []
        )

    def reset(self, key_random: jnp.ndarray) -> StateSpeciesAdaRL:

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

        return StateSpeciesAdaRL(
            agents=batch_agents,
            metrics_lifespan=list_metrics_lifespan,
            metrics_population=list_metrics_population,
        )

    def init_hp(self) -> HyperParametersAdaRL:
        """Get the initial hyperparameters of the agent from the config"""
        return HyperParametersAdaRL(**self.config["hp_initial"])

    def init_agent(
        self,
        key_random: jnp.ndarray,
        do_exist: bool = True,
        params: Optional[Dict[str, jnp.ndarray]] = None,
        hp: Optional[HyperParametersAdaRL] = None,
        table_value_fruits: Optional[jnp.ndarray] = None,
        table_value_fruits_initial: Optional[jnp.ndarray] = None,
        params_reward: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> AgentAdaRL:
        """Create a new agent.

        Args:
            key_random (jnp.ndarray): the random key
            do_exist (bool, optional): whether the agent exists. Defaults to True.
            params (Optional[Dict[str, jnp.ndarray]], optional): the parameters of the agent. Defaults to None.
            hp (Optional[HyperParametersAdaRL], optional): the hyperparameters of the agent. Defaults to None.
            table_value_fruits (Optional[jnp.ndarray], optional): the table of values of the fruits. Defaults to None.
            table_value_fruits_initial (Optional[jnp.ndarray], optional): the initial table of values of the fruits. Defaults to None.
            params_reward (Optional[Dict[str, jnp.ndarray]], optional): the parameters of the reward model. Defaults to None.
        """
        if params is None:
            key_random, subkey = random.split(key_random)
            variables = self.model.get_initialized_variables(subkey)
            params = variables.get("params", {})
        if hp is None:
            hp = self.init_hp()
        if table_value_fruits is None:
            table_value_fruits = jnp.zeros((4,))
        if (
            table_value_fruits_initial is None
            and self.mode_weights_transmission == "initial"
        ):
            table_value_fruits_initial = table_value_fruits
        if params_reward is None:
            key_random, subkey = random.split(key_random)
            variables = self.reward_model.get_initialized_variables(subkey)
            params_reward = variables.get("params", {})
        key_random, subkey = random.split(key_random)
        obs_dummy = self.space_observation_fruit_averager.sample(subkey)

        return AgentAdaRL(
            age=0,
            params=params,
            hp=hp,
            table_value_fruits=table_value_fruits,
            table_value_fruits_initial=table_value_fruits_initial,
            params_reward=params_reward,
            do_exist=do_exist,
            obs_last=obs_dummy,  # dummy observation
            action_last=-1,  # dummy action
        )

    def react(
        self,
        state: StateSpeciesAdaRL,
        batch_observations: ObservationAgent,  # Batched
        eco_information: EcoInformation,
        key_random: jnp.ndarray,
    ) -> Tuple[
        StateSpeciesAdaRL,
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

        new_agents: AgentAdaRL = tree_map(
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

        new_state: StateSpeciesAdaRL = state.replace(agents=new_agents)

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

    def mutate_state_agent(
        self, agent: AgentAdaRL, key_random: jnp.ndarray
    ) -> AgentAdaRL:

        # Mutate the hyperparameters
        key_random, *subkeys = random.split(key_random, 5)
        new_hp = HyperParametersAdaRL(
            strength_mutation=mutate_scalar(
                value=agent.hp.strength_mutation, range=(0, None), key_random=subkeys[3]
            ),
        )

        # Mutate the weights, slightly mutated
        key_random, subkey = random.split(key_random)
        new_params = mutation_gaussian_noise(
            arr=agent.params,
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

        # Transmit (or not) the table_value_fruits according to the mode
        if self.mode_weights_transmission == "initial":
            # Mute and transmit the initial table_value_fruits
            key_random, subkey = random.split(key_random)
            new_table_value_fruits_initial = mutation_gaussian_noise(
                arr=agent.table_value_fruits_initial,
                strength_mutation=agent.hp.strength_mutation,
                key_random=subkey,
            )
            new_table_value_fruits = new_table_value_fruits_initial
        elif self.mode_weights_transmission == "final":
            # Transmit the final table_value_fruits
            new_table_value_fruits = agent.table_value_fruits
            new_table_value_fruits_initial = None
        elif self.mode_weights_transmission == "none":
            # Don't transmit any table_value_fruits
            new_table_value_fruits = None
            new_table_value_fruits_initial = None
        else:
            raise NotImplementedError(
                f"Mode {self.mode_weights_transmission} not implemented"
            )

        return self.init_agent(
            key_random=key_random,
            params=new_params,
            hp=new_hp,
            table_value_fruits=new_table_value_fruits,
            table_value_fruits_initial=new_table_value_fruits_initial,
            params_reward=new_params_reward,
        )

    # =============== Agent inference & learning methods =================

    def react_agents(
        self,
        key_random: jnp.ndarray,
        state: StateSpeciesAdaRL,
        batch_observations: ObservationAgent,  # Batched
    ) -> Tuple[StateSpeciesAdaRL, ActionAgent]:  # Batched

        def react_single_agent(
            key_random: jnp.ndarray,
            agent: AgentAdaRL,
            obs: jnp.ndarray,
        ) -> Tuple[
            AgentAdaRL,
            int,
            Dict[str, jnp.ndarray],
        ]:
            # =============== Inference part =================
            key_random, subkey = random.split(key_random)
            obs["table_value_fruits"] = agent.table_value_fruits
            logits = self.model.apply(
                variables={"params": agent.params},
                x=obs,
                key_random=subkey,
            )
            key_random, subkey = random.split(key_random)
            action = random.categorical(key_random, logits=logits)

            # ============== Learning part =================

            # Compute the reward : r_t = reward_model(o_t, o_{t+1})
            key_random, subkey = random.split(key_random)
            reward_last = self.reward_model.apply(
                variables={"params": agent.params_reward},
                x={"obs": agent.obs_last, "obs_next": obs},
                key_random=subkey,
            )
            reward_last *= (
                agent.age > 0
            )  # If agent is just born, don't learn (reward set at 0)

            # Update the table of values of the fruits
            idx_eat_action = {
                name_action: idx_action
                for idx_action, name_action in self.env.action_idx_to_meaning().items()
            }["eat"]
            idx_center = obs["visual_field"].shape[0] // 2
            condlist = []
            choicelist = []
            for id_fruit in range(4):
                idx_channel_fruit = self.list_channels_visual_field.index(
                    f"fruits_{id_fruit}"
                )
                condlist.append(
                    (
                        agent.obs_last["visual_field"][
                            idx_center, idx_center, idx_channel_fruit
                        ]
                        == 1.0
                    )
                    & (agent.action_last == idx_eat_action)
                )
                choicelist = agent.table_value_fruits.at[id_fruit].set(reward_last)
            table_value_fruits_new = jnp.select(
                condlist=condlist,
                choicelist=choicelist,
                default=agent.table_value_fruits,
            )

            # Update the agent's state
            agent = agent.replace(
                age=agent.age + 1,
                table_value_fruits=table_value_fruits_new,
                obs_last=obs,
                action_last=action,
            )

            # ============== Measures ==============
            dict_measures = {}
            if "reward" in self.names_measures:
                dict_measures["reward"] = reward_last
            if "table_value_fruits" in self.names_measures:
                for id_fruit in range(4):
                    dict_measures[f"table_value_fruits_{id_fruit}"] = (
                        agent.table_value_fruits[id_fruit]
                    )

            # Update the agent's state and act
            return agent, action, dict_measures

        batch_keys = random.split(key_random, self.n_agents_max)
        new_agents, actions, dict_measures = jax.vmap(react_single_agent)(
            key_random=batch_keys,
            agent=state.agents,
            obs=batch_observations,
        )
        dict_measures.update(**new_agents.params_reward)

        new_state = state.replace(
            agents=new_agents,
        )
        return new_state, actions, dict_measures

    # =============== Metrics methods =================

    def render(self, state: StateSpeciesAdaRL, force_render: bool = False) -> None:
        """Do the rendering of the species. This can be a visual rendering or a logging of the state of any kind.

        Args:
            state (StateSpecies): the state of the species to render
            force_render (bool): whether to force the rendering even if the species is not in a state where it should be rendered
        """
        return
        # Log heatmaps of the weights
        try:
            weights = state.agents.params_decision["Dense_0"]["kernel"].mean(axis=0)
            n_obs, n_actions = weights.shape
            bias = state.agents.params_decision["Dense_0"]["bias"].mean(axis=0)
            plt.figure(figsize=(10, 8))
            sns.heatmap(weights, annot=True, cmap="viridis", cbar=True)
            plt.xlabel("Actions")
            plt.ylabel("Observations")
            plt.title("Heatmap of Weights")
            os.makedirs("logs/heatmaps", exist_ok=True)
            plt.savefig(f"logs/heatmaps/heatmap.png")

            x = np.arange(n_actions)  # the label locations
            width = 0.35  # the width of the bars
            sum_weights = np.sum(weights, axis=0)

            fig, ax = plt.subplots(figsize=(10, 6))
            bars1 = ax.bar(x - width / 2, bias, width, label="Bias", color="skyblue")
            bars2 = ax.bar(
                x + width / 2,
                sum_weights,
                width,
                label="Sum of Weights",
                color="lightgreen",
            )

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_xlabel("Actions")
            ax.set_ylabel("Values")
            ax.set_title("Bias and Sum of Weights for Each Action")
            ax.set_xticks(x)
            ax.set_xticklabels(x)
            ax.legend()

            # Path to save the combined bar chart
            combined_bar_chart_path = os.path.join(
                "logs/heatmaps/bias and sum weights.png"
            )

            # Save the combined bar chart
            plt.savefig(combined_bar_chart_path)
            plt.close()

        except Exception as e:
            print(f"Error in agents render: {e}")

    def compute_measures(
        self,
        state: StateSpeciesAdaRL,
        state_new: StateSpeciesAdaRL,
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
            if "hp" == name_measure:
                strength_mutation = getattr(state.agents.hp, "strength_mutation")
                dict_measures["hp strength_mutation"] = strength_mutation
                dict_measures["hp log10_strength_mutation"] = jnp.log10(
                    strength_mutation
                )
            if name_measure == "params_reward_model":
                for name_param, values_param in state.agents.params_reward.items():
                    dict_measures[f"params_reward_model {name_param}"] = values_param
            if name_measure == "weights_agents":
                continue
                # Metric for logging the weights between Gridworld env and the first layer of the neural network
                try:
                    weights = state.agents.params_decision["Dense_0"]["kernel"]
                    bias = state.agents.params_decision["Dense_0"]["bias"]

                    # _, n_obs, n_actions = weights.shape
                    # for idx_action in range(n_actions):
                    #     dict_measures[f"weights b{idx_action}/weights"] = bias[:, idx_action].mean()
                    #     for idx_obs in range(n_obs):
                    #         dict_measures[f"weights w{idx_obs}-{idx_action}/weights"] = weights[:, idx_obs, idx_action].mean()

                    action_idx_to_meaning = self.env.action_idx_to_meaning()
                    obs_idx_to_meaning = self.env.obs_idx_to_meaning()
                    for idx_action, name_action in action_idx_to_meaning.items():
                        dict_measures[f"weights bias{name_action}/weights"] = bias[
                            :, idx_action
                        ]
                        for idx_obs, name_obs in obs_idx_to_meaning.items():
                            dict_measures[
                                f"weights {name_obs}-{name_action}/weights"
                            ] = weights[:, idx_obs, idx_action]

                except Exception as e:
                    print(f"Error in measure weights_agents: {e}")
            # Behavior measures
            pass

        return dict_measures
