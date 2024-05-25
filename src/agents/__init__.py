from typing import Any, Dict, Type, Generic, TypeVar

from flax import struct

from src.agents.base_agent_species import BaseAgentSpecies
from src.agents.neuro_evolution import NeuroEvolutionAgentSpecies



    

agent_name_to_AgentSpeciesClass: Dict[str, Type[NeuroEvolutionAgentSpecies]] = {
    "NeuroEvolution Agents": NeuroEvolutionAgentSpecies,  # temp typing here
}
