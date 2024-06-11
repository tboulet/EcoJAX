from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Type, Union



class BaseLogger(ABC):
    """Base class for all loggers"""
    
    @abstractmethod
    def log_scalars(self, key: str, dict_scalars: Dict[str, float], timestep: int):
        """Log dictionary of scalars"""
        raise NotImplementedError
    
    @abstractmethod
    def log_histograms(self, key: str, dict_histograms: Dict[str, List[float]], timestep: int):
        """Log dictionary of histograms"""
        raise NotImplementedError
    
    @abstractmethod
    def close(self):
        """Close the logger"""
        raise NotImplementedError