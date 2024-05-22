from typing import Dict, Type

from src.agents.base_agent_species import BaseAgentSpecies
from src.agents.neuro_evolution import NeuroEvolutionAgentSpecies

agent_name_to_AgentSpeciesClass : Dict[str, Type[NeuroEvolutionAgentSpecies]] = {  # temp typing here
    "NeuroEvolution Agents" : NeuroEvolutionAgentSpecies,
}