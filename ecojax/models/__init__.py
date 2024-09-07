from typing import Dict, Type

from ecojax.models.acnn import AdaptedCNN_Model
from ecojax.models.base_model import BaseModel
from ecojax.models.cnn import CNN_Model
from ecojax.models.human import HumanModel
from ecojax.models.random import RandomModel
from ecojax.models.mlp import MLP_Model
from ecojax.models.reactive import ReactiveModel
from ecojax.models.region import RegionalModel

model_name_to_ModelClass: Dict[str, Type[BaseModel]] = {
    "Random Model": RandomModel,
    "ReactiveModel" : ReactiveModel,
    "HumanModel": HumanModel,
    "MLP": MLP_Model,
    "CNN": CNN_Model,
    "AdaptedCNN": AdaptedCNN_Model,
    "RegionalModel": RegionalModel,
}
