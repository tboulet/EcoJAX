from typing import Dict, Type

from ecojax.models.base_model import BaseModel
from ecojax.models.cnn import CNN_Model
from ecojax.models.random import RandomModel
from ecojax.models.mlp import MLP_Model
from ecojax.models.reactive import ReactiveModel

model_name_to_ModelClass: Dict[str, Type[BaseModel]] = {
    "Random Model": RandomModel,
    "ReactiveModel" : ReactiveModel,
    "MLP": MLP_Model,
    "CNN": CNN_Model,
}
