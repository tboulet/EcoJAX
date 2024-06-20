from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Type, Union

from ecojax.core import EcoInformation


class BaseLogger(ABC):
    """Base class for all loggers"""

    @abstractmethod
    def log_scalars(
        self,
        dict_scalars: Dict[str, float],
        dict_reproduction: Dict[int, List[int]],
        timestep: int,
    ):
        """Log dictionary of scalars"""
        raise NotImplementedError

    @abstractmethod
    def log_histograms(
        self,
        dict_histograms: Dict[str, List[float]],
        timestep: int,
    ):
        """Log dictionary of histograms"""
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """Close the logger"""
        raise NotImplementedError

    def log_eco_metrics(
        self,
        eco_information : EcoInformation,
        timestep: int,
    ):
        """Log ecp metrics"""
        pass
