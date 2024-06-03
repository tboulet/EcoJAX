from typing import Dict, Type

from src.models.base_model import BaseModel
from src.models.random import RandomModel
from src.models.mlp import MLP_Model

model_name_to_ModelClass : Dict[str, Type[BaseModel]] = {
    "MLP" : MLP_Model,
    "Random Model" : RandomModel,
}