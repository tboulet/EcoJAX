from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np


class BaseAgentSpecies(ABC):

    def __init__(self, config: Dict):
        self.config = config

    @abstractmethod
    def react(self, observations: Dict) -> Dict:
        pass
