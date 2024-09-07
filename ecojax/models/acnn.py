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
from ecojax.utils import add_scalars_as_channels_single, average_pooling


class AdaptedCNN_Model(BaseModel):
    """A model adapted to the gridworld environment that is able to use a CNN to process visual field and other scalar observations.

    Only works with observations that are a dict containing a visual field of shape (h,h,C) where h is a multiple of 3, and of other m scalar observations.

    It does the following :
            - for each m scalar observations, add it as a channel to the visual field
            - on the (h,h,C+m) tensor, apply a CNN to obtain a (h,h,C') tensor
            - apply global pooling to reduce dimension on external pixels
            - flatten the tensor to obtain a (9, C') tensor
            - add the center pixel to the tensor to obtain a (10, C') tensor
            - apply a MLP to obtain a (n_output_features,) tensor

    Args:
        cnn_config (Dict[str, Any): the configuration of the CNN(s). This config is (for now) common to all the CNNs. It should contain the following :
            - hidden_dims (List[int]): the number of hidden units in each hidden layer, it also define therefore the number of hidden layers
            - kernel_size (int) : the kernel_size of the CNN(s)
            - strides (int) : the stride of the CNN(s)
        mlp_pixel_config (Dict[str, Any]): the configuration of the MLP that will be applied to each pixel. It should contain the following :
            - hidden_dims (List[int]): the number of hidden units in each hidden layer, it also define therefore the number of hidden layers
            - n_output_features (int): the number of output features of the MLP (should be 1)
        mlp_config (Dict[str, Any]): the configuration of the MLP. It should contain the following :
            - hidden_dims (List[int]): the number of hidden units in each hidden layer, it also define therefore the number of hidden layers
            - n_output_features (int): the number of output features of the MLP

    Improvements :
        - instead of using a CNN, maybe average across pixel of the same direction (forward, backward, left, right, center) to obtain a (5, C+m) tensor, eventually scaled by proximity to the center
    """

    cnn_config: Dict[str, Any]
    range_nearby_pixels: int
    mlp_pixel_config: Dict[str, Any]
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
        assert S % 3 == 0, "The visual field must have a size that is a multiple of 3"
                
        # Add the scalar observations as channels to the visual field
        visual_field = obs["visual_field"]
        list_values = []
        for key, value in obs.items():
            if key != "visual_field":
                shape_value = value.shape
                if len(shape_value) == 0:
                    list_values.append(value)
                elif len(shape_value) == 1:
                    list_values.extend(value)
                else:
                    raise ValueError(
                        f"Only visual field and scalar observations are supported, but got {key} with shape {shape_value}"
                    )
                    
        visual_field_with_scalars = add_scalars_as_channels_single(
            image=visual_field, scalars=jnp.array(list_values)
        ) # (S, S, C+m)
        C_visual_field_with_scalars = C_visual_field + len(list_values)
        
        # Apply the CNN
        if len(self.cnn_config["hidden_dims"]) > 0:
            C_output_cnn = self.cnn_config["hidden_dims"][-1]
            map_post_convolution = CNN(**self.cnn_config, shape_output=(S, S, C_output_cnn))(
                visual_field_with_scalars
            ) # (S, S, C')
        else:
            C_output_cnn = C_visual_field_with_scalars
            map_post_convolution = visual_field_with_scalars # (S, S, C'=C+m)
        
        # Apply global pooling
        side_length = S // 3
        map_post_pooling = average_pooling(input_array=map_post_convolution, h=side_length)
        x = jnp.reshape(map_post_pooling, (-1, C_output_cnn)) # (9, C')
        
        # Add nearby pixels
        idx_center = S // 2
        R = side_length // 2
        R = min(R, self.range_nearby_pixels)
        list_pixels = []
        for i in range(idx_center - R, idx_center + R + 1):
            for j in range(idx_center - R, idx_center + R + 1):
                list_pixels.append(jnp.expand_dims(map_post_convolution[i, j, :], axis=0))
        x = jnp.concatenate([x] + list_pixels, axis=0) # (9 + R^2, C')
        
        # Apply dense layers across the flattened tensor
        assert self.mlp_pixel_config["n_output_features"] == 1, "The number of output features of the pixel MLP should be 1"
        x = MLP(**self.mlp_pixel_config)(x) # (9 + R^2, 1)
        x = x[:, 0] # (9 + R^2,)
    
        # Apply the MLP and return the output
        x = MLP(**self.mlp_config)(x)
        return x
