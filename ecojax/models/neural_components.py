from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Union
import numpy as np

import jax
from jax import random
import jax.numpy as jnp
from flax import struct
import flax.linen as nn

from ecojax.models.base_model import BaseModel
from ecojax.types import ObservationAgent, ActionAgent
from ecojax.spaces import ContinuousSpace, DiscreteSpace
from ecojax.utils import jprint, jprint_and_breakpoint


names_activations_to_fn: Dict[str, Callable[[jnp.ndarray], jnp.ndarray]] = {
    "linear": lambda x: x,
    "relu": nn.relu,
    "tanh": nn.tanh,
    "sigmoid": nn.sigmoid,
    "leaky_relu": nn.leaky_relu,
    "softplus": nn.softplus,
    "swish": nn.swish,
}


class MLP(nn.Module):
    """
    A simple MLP model. It will create a MLP that does the following inference :
    MLP : input -> hidden_dims[0], -> hidden_dims[1], -> ... -> hidden_dims[-1], -> (n_output_features,)

    Args:
        hidden_dims (List[int]): the number of hidden units in each hidden layer. Also defines the number of hidden layers.
        n_output_features (int): the number of output features
        name_activation_fn (str, optional): the name of the activation function. Defaults to "swish".
        name_activation_output_fn (str, optional): the name of the activation function for the output. Defaults to "linear".
    """

    hidden_dims: List[int]
    n_output_features: int
    name_activation_fn: str = "swish"
    name_activation_output_fn: str = "swish"

    def setup(self) -> None:
        self.activation_fn = names_activations_to_fn[self.name_activation_fn]
        self.activation_output_fn = names_activations_to_fn[
            self.name_activation_output_fn
        ]
        super().setup()

    @nn.compact
    def __call__(self, x):
        """Forward pass of the MLP.

        Args:
            x (jnp.ndarray): the input data

        Returns:
            jnp.ndarray: the output data
        """
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(features=hidden_dim)(x)
            x = self.activation_fn(x)
        x = nn.Dense(features=self.n_output_features)(x)
        x = self.activation_output_fn(x)
        return x


class CNN(nn.Module):
    """
    A CNN model adapted to the input and output shapes.

    Args:
        hidden_dims (List[int]): the number of hidden units in each hidden layer. Also defines the number of hidden layers.
        kernel_size (List[int]): the size of the kernel, common to all layers
        strides (List[int]): the size of the strides, common to all layers
        shape_output (List[int]): the shape of the output
        name_activation_fn (str, optional): the name of the activation function. Defaults to "relu".
        name_activation_output_fn (str, optional): the name of the activation function for the output. Defaults to "linear".
    """

    hidden_dims: List[int]
    kernel_size: List[int]
    strides: List[int]
    shape_output: List[int]
    name_activation_fn: str = "swish"
    name_activation_output_fn: str = "swish"

    def setup(self) -> None:
        self.activation_fn = names_activations_to_fn[self.name_activation_fn]
        self.activation_output_fn = names_activations_to_fn[
            self.name_activation_output_fn
        ]
        super().setup()

    @nn.compact
    def __call__(self, x):
        """Forward pass of the CNN.

        Args:
            x (jnp.ndarray): the input data

        Returns:
            jnp.ndarray: the output data
        """
        assert len(x.shape) in [
            2,
            3,
        ], f"Expected input shape to be 2 or 3 but got {len(x.shape)}"

        # Convert the shape from (C, H, W) to (H, W, C) if needed,
        # i.e. when there is three dimensions with the first one being smaller than the other two
        if len(x.shape) == 3 and x.shape[0] < x.shape[1] and x.shape[0] < x.shape[2]:
            x = jnp.transpose(x, (1, 2, 0))
        H, W = x.shape[:2]

        # Apply the CNN : (H, W, C) -> (H, W, hidden_dim) -> ... -> (H, W, hidden_dim)
        for hidden_dim in self.hidden_dims:
            x = nn.Conv(
                features=hidden_dim,
                kernel_size=(self.kernel_size, self.kernel_size),
                strides=(self.strides, self.strides),
            )(x)
            x = self.activation_fn(x)

        # Apply the output layer depending on the shape of the output
        if len(self.shape_output) == 0:
            # If scalar, do the average of the last layer : (H, W, C) -> mean -> ()
            x = jnp.mean(x)
        elif len(self.shape_output) == 1:
            # If embedding (n,), apply conv and then dense : (H, W, C) -> conv -> (H, W, 1) -> reshape -> (H * W,) -> dense -> (n,)
            x = nn.Conv(
                features=1,
                kernel_size=(1, 1),
                strides=(1, 1),
            )(x)
            x = jnp.reshape(x, (-1,))
            x = nn.Dense(features=self.shape_output[0])(x)
        elif len(self.shape_output) == 2:
            # Shape (H, W) : For now only having the desired shape corresponding to the dimension of the input (H, W) is supported
            # Assert the shape correspond to (H, W) and apply a mean accross the channel dimension
            assert self.shape_output == [
                H,
                W,
            ], f"Expected shape_output to be {H, W} but got {self.shape_output}"
            x = jnp.mean(x, axis=-1)
        elif len(self.shape_output) == 3:
            # Shape (H, W, C') : For now only having the (H, W) corresponding to the dimension of the input is supported
            # Assert the shape correspond to (H, W) and apply a convolutional layer to get the right shape
            H_output, W_output, C_output = self.shape_output
            assert (
                H_output == H and W_output == W
            ), f"Expected shape_output to be {H, W, '?'} but got {self.shape_output}"
            H, W, C_last = x.shape
            if C_last != C_output:
                # Apply a convolutional layer to get the right shape
                x = nn.Conv(
                    features=C_output,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                )(x)
            else:
                # Simply return the value
                x = x

        # Apply the output activation function and return the output
        x = self.activation_output_fn(x)
        return x
