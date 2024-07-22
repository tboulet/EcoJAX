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
from ecojax.spaces import ContinuousSpace, DiscreteSpace


class CNN_Model(BaseModel):
    """A model that is able to use a CNN to process image-like observations.
    It does the following :
    - process the image-like observations with CNN(s) to obtain outputs of shape (dim_cnn_output,)
    - concatenate the outputs of the CNN(s) and other observation components to obtain a single vector
    - process the concatenated output with a final MLP to obtain an output of shape (n_output_features,)
    - generate the outputs for each action space through finals MLPs

    Args:
        cnn_config (Dict[str, Any): the configuration of the CNN(s). This config is (for now) common to all the CNNs. It should contain the following :
            - hidden_dims (List[int]): the number of hidden units in each hidden layer, it also define therefore the number of hidden layers
            - kernel_size (int) : the kernel_size of the CNN(s)
            - strides (int) : the stride of the CNN(s)
        dim_cnn_output (int) : the dimension of the output of the CNN(s)
        mlp_config (Dict[str, Any]): the configuration of the MLP. It should contain the following :
            - hidden_dims (List[int]): the number of hidden units in each hidden layer, it also define therefore the number of hidden layers
            - n_output_features (int): the number of output features of the MLP
    """

    cnn_config: Dict[str, Any]
    dim_cnn_output: int
    mlp_config: Dict[str, Any]

    def obs_to_encoding(
        self, obs: ObservationAgent, key_random: jnp.ndarray
    ) -> jnp.ndarray:

        list_spaces_and_values = self.space_input.get_list_spaces_and_values(obs)
        list_vectors = []
        for (space, x) in list_spaces_and_values:
            if isinstance(space, ContinuousSpace):
                n_dim = len(space.shape)
                if n_dim == 0:
                    encoding = jnp.expand_dims(x, axis=-1)
                elif n_dim == 1:
                    encoding = x
                elif n_dim == 2:
                    encoding = CNN(**self.cnn_config, shape_output=(self.dim_cnn_output,))(x)
                elif n_dim == 3:
                    encoding = CNN(**self.cnn_config, shape_output=(self.dim_cnn_output,))(x)
                else:
                    raise ValueError(
                        f"Continuous observation spaces with more than 3 dimensions are not supported"
                    )
            elif isinstance(space, DiscreteSpace):
                encoding = jax.nn.one_hot(x, space.n)
            else:
                raise ValueError(f"Unknown space type for observation: {type(space)}")
            
            list_vectors.append(encoding)

        # Concatenate the latent vectors
        x = jnp.concatenate(list_vectors, axis=-1)

        # Define the hidden layers of the MLP
        x = MLP(**self.mlp_config)(x)

        # Return the action and the probability of the action
        return x
