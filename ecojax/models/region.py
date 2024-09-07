from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
import numpy as np

import jax
from jax import random
import jax.numpy as jnp
from flax import struct
import flax.linen as nn

from ecojax.models.base_model import BaseModel
from ecojax.models.neural_components import CNN, MLP
from ecojax.types import ObservationAgent, ActionAgent
from ecojax.spaces import ContinuousSpace, DictSpace, DiscreteSpace
from ecojax.utils import add_scalars_as_channels_single, separate_visual_field


class RegionalModel(BaseModel):
    """A model adapted to the gridworld environment that will separate the visual field into 5 regions (center, up, down, left, right) and compute the (weighted) average of each region.
    
    It does the following :
        - for each m scalar observations, add it as a channel to the visual field
        - separate the visual field into 5 regions (center, up, down, left, right)
            Here it possibly makes sense to weight the pixels by their proximity to the center
        - for each region, we obtain a (C+m,), so in total a (5, C+m) tensor
        - apply an MLP to each region to obtain a (5, 1) tensor
        - squeeze and add a final MLP to obtain a (n_output_features,) tensor
        
    Args:
        weighting_method (str): the method to weight the pixels. For now, only "uniform" is implemented
        mlp_region_config (Dict[str, Any]): the configuration of the MLP that will be applied to each region. It should contain the following :
            - hidden_dims (List[int]): the number of hidden units in each hidden layer, it also define therefore the number of hidden layers
            - n_output_features (int): the number of output features of the MLP
        mlp_config (Dict[str, Any]): the configuration of the final MLP. It should contain the following :
            - hidden_dims (List[int]): the number of hidden units in each hidden layer, it also define therefore the number of hidden layers
            - n_output_features (int): the number of output features of the MLP
    """

    weighting_method: str
    mlp_region_config: Dict[str, Any]
    mlp_config: Dict[str, Any]
    
    def obs_to_encoding(
        self, obs: ObservationAgent, key_random: jnp.ndarray
    ) -> jnp.ndarray:
        assert isinstance(
            self.space_input, DictSpace
        ), "The input space must be a DictSpace"
        assert "visual_field" in obs, "The observation must contain a visual field"

        S, S_, C_visual_field = obs["visual_field"].shape
        assert S == S_, "The visual field must be a square"

        # Add the scalar observations as channels to the visual field
        visual_field = obs["visual_field"]
        list_values = []
        for key, value in obs.items():
            if key != "visual_field":
                shape_value = value.shape
                if len(shape_value) == 0:
                    list_values.append(value)
                elif len(shape_value) == 1:
                    list_values.extend([value[i] for i in range(shape_value[0])])
                else:
                    raise ValueError(
                        f"Only visual field and scalar observations are supported, but got {key} with shape {shape_value}"
                    )
                    
        visual_field_with_scalars = add_scalars_as_channels_single(
            image=visual_field, scalars=jnp.array(list_values)
        ) # (S, S, C+m)
        
        # Obtain the regions
        if self.weighting_method == "uniform":
            regions = separate_visual_field(visual_field_with_scalars) # (5, C+m)
        else:
            raise NotImplementedError(f"Weighting method {self.weighting_method} is not implemented")
        
        # Apply the MLP for each region
        assert self.mlp_region_config["n_output_features"] == 1, "The MLP for the regions should have a single output feature"
        x = MLP(**self.mlp_region_config)(regions) # (5, 1)
        x = x[:, 0] # (5,)
        
        # Apply a final MLP
        x = MLP(**self.mlp_config)(x)

        # Return the output
        return x
