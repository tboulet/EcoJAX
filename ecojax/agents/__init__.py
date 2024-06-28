from typing import Any, Dict, Type, Generic, TypeVar

from flax import struct

from ecojax.agents.base_agent_species import AgentSpecies
from ecojax.agents.neuro_evolution import NeuroEvolutionAgentSpecies
from ecojax.agents.reinforcement_learning import RL_AgentSpecies

agent_name_to_AgentSpeciesClass: Dict[str, Type[AgentSpecies]] = {
    "NeuroEvolutionAgents": NeuroEvolutionAgentSpecies,
    "RL_Agents": RL_AgentSpecies,
}
