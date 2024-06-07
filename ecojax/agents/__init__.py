from typing import Any, Dict, Type, Generic, TypeVar

from flax import struct

from ecojax.agents.base_agent_species import BaseAgentSpecies
from ecojax.agents.neuro_evolution import NeuroEvolutionAgentSpecies


agent_name_to_AgentSpeciesClass: Dict[str, Type[BaseAgentSpecies]] = {
    "NeuroEvolution Agents": NeuroEvolutionAgentSpecies,
}
