from typing import Any, Dict, Type, Generic, TypeVar

from flax import struct

from src.agents.base_agent_species import BaseAgentSpecies

from src.agents.neuro_evolution import NeuroEvolutionAgentSpecies
from src.agents.agents_random import RandomAgentSpecies


    

agent_name_to_AgentSpeciesClass: Dict[str, Type[BaseAgentSpecies]] = {
    "Random Agents": RandomAgentSpecies,
    "NeuroEvolution Agents": NeuroEvolutionAgentSpecies,
}
