from enum import Enum
from typing import List



class CategoryMeasures(Enum):
    """The different categories of measures that can be computed in the metrics module.
    """
    IMMEDIATE = "immediate"
    STATE = "state"
    LIFESPAN = "lifespan"