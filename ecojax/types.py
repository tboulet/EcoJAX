from abc import ABC, abstractmethod
from flax import struct


@struct.dataclass
class StateEnv:
    """This JAX data class represents the state of the environment. It is used to store the state of the environment and to apply JAX transformations to it.
    Instances of this class represents objects that will change of value through the simulation and that entirely representing the non-constant part of the environment.
    """


@struct.dataclass
class ObservationAgent:
    """This class represents the observation of an agent. It is used to store the observation of an agent and to apply JAX transformations to it.
    Instancess of this class represents objects that will be given as input to the agents by the environment.
    """

@struct.dataclass
class ActionAgent:
    """This class represents the action of an agent. It is used to store the action of an agent and to apply JAX transformations to it.
    Instances of this class represents objects that will be output by the agents and given as input to the environment.
    """
    
    
@struct.dataclass
class StateAgent:
    """This class represents the current state of an agent. It is used to store the state of an agent and to apply JAX transformations to it.
    Instances of this class represents objects that will change of value through the simulation and that entirely representing the non-constant part of the agent.
    """
    
    
class PytreeLike(ABC):
    """This class represents a Pytree-like object. It is used to store a Pytree-like object and to apply JAX transformations to it.
    """
    
    @abstractmethod
    def tree_flatten(self):
        """Flatten the Pytree-like object.
        
        Returns:
            leaves (Tuple): the flattened Pytree-like object
            aux_data (Any): the auxiliary data
        """
        raise NotImplementedError
    
    @classmethod
    @abstractmethod
    def tree_unflatten(cls, aux_data, leaves):
        """Unflatten the Pytree-like object.
        
        Args:
            aux_data (Any): the auxiliary data
            leaves (Tuple): the flattened Pytree-like object
        
        Returns:
            PytreeLike: the unflattened Pytree-like object
        """
        raise NotImplementedError
    
    

    