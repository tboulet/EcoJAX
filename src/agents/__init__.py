from typing import Any, Dict, Type, Generic, TypeVar

from flax import struct

from src.agents.base_agent_species import BaseAgentSpecies
from src.agents.neuro_evolution import NeuroEvolutionAgentSpecies


@struct.dataclass
class EnvState:
    """This JAX data class represents the state of the environment. It is used to store the state of the environment and to apply JAX transformations to it.
    Instances of this class represents objects that will change of value through the simulation and that entirely representing the non-constant part of the environment.
    """


@struct.dataclass
class AgentObservation:
    """This class represents the observation of an agent. It is used to store the observation of an agent and to apply JAX transformations to it.
    Instancess of this class represents objects that will be given as input to the agents.
    """
    

agent_name_to_AgentSpeciesClass: Dict[str, Type[NeuroEvolutionAgentSpecies]] = {
    "NeuroEvolution Agents": NeuroEvolutionAgentSpecies,  # temp typing here
}
