from typing import Dict, Type

from ecojax.models.base_model import BaseModel
from ecojax.models.random import RandomModel
from ecojax.models.mlp import MLP_Model

model_name_to_ModelClass: Dict[str, Type[BaseModel]] = {
    "MLP": MLP_Model,
    "Random Model": RandomModel,
}
