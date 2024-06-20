from abc import ABC, abstractmethod
from typing import Dict, List, Union
import numpy as np

import jax
from jax import random
import jax.numpy as jnp
from flax import struct
import flax.linen as nn

from ecojax.models.base_model import BaseModel
from ecojax.types import ObservationAgent, ActionAgent
from ecojax.spaces import Continuous, Discrete


names_activations_to_fn = {
    "linear": lambda x: x,
    "relu": nn.relu,
    "tanh": nn.tanh,
    "sigmoid": nn.sigmoid,
    "leaky_relu": nn.leaky_relu,
    "softplus": nn.softplus,
}


class MLP(nn.Module):

    def __init__(
        self,
        hidden_dims: List[int],
        n_output_features: int,
        name_activation_fn: str = "relu",
        name_activation_output_fn: str = "linear",
    ):
        """Create a simple MLP model.

        Args:
            hidden_dims (List[int]): the number of hidden units in each hidden layer
            n_output_features (int): the number of output features
            name_activation_fn (str, optional): the name of the activation function. Defaults to "relu".
        """
        self.hidden_dims = hidden_dims
        self.output_features = n_output_features
        self.activation_fn = names_activations_to_fn[name_activation_fn]
        self.activation_output_fn = names_activations_to_fn[name_activation_output_fn]

    @nn.compact
    def __call__(self, x):
        """Forward pass of the MLP.

        Args:
            x (jnp.ndarray): the input data

        Returns:
            jnp.ndarray: the output data
        """
        x = nn.Dense(features=self.hidden_dims[0])(x)
        x = self.activation_fn(x)
        for hidden_dim in self.hidden_dims[1:]:
            x = nn.Dense(features=hidden_dim)(x)
            x = self.activation_fn(x)
        x = nn.Dense(features=self.output_features)(x)
        x = self.activation_output_fn(x)
        return x


class CNN(nn.Module):

    def __init__(
        self,
        hidden_dims: List[int],
        kernel_size: List[int],
        strides: List[int],
        shape_output: List[int],
        name_activation_fn: str = "relu",
        name_activation_output_fn: str = "linear",
    ):
        """Create a simple CNN model.

        Args:
            hidden_dims (List[int]): the number of hidden units in each hidden layer
            shape_output (List[int]): the shape of the output
            name_activation_fn (str, optional): the name of the activation function. Defaults to "relu".
            name_activation_output_fn (str, optional): the name of the activation function for the output. Defaults to "linear".
        """
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        self.strides = strides
        self.shape_output = shape_output
        self.activation_fn = names_activations_to_fn[name_activation_fn]
        self.activation_output_fn = names_activations_to_fn[name_activation_output_fn]

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

        # Apply the CNN
        x = nn.Conv(
            features=self.hidden_dims[0],
            kernel_size=self.kernel_size,
            strides=self.strides,
        )(x)
        x = self.activation_fn(x)
        for hidden_dim in self.hidden_dims[1:]:
            x = nn.Conv(
                features=hidden_dim, kernel_size=self.kernel_size, strides=self.strides
            )(x)
            x = self.activation_fn(x)

        # Apply the output layer depending on the shape of the output
        if len(self.shape_output) == 0:
            # Do the average of the last layer
            x = jnp.mean(x)
        elif len(self.shape_output) == 1:
            # Flatten the output and apply a dense layer
            x = nn.Flatten()(x)
            x = nn.Dense(features=self.shape_output[0])(x)
        elif len(self.shape_output) == 2:
            # Assert the shape correspond to (H, W)
            assert self.shape_output == [
                H,
                W,
            ], f"Expected shape_output to be {H, W} but got {self.shape_output}"
            # Average along the n_features dimension
            x = jnp.mean(x, axis=-1)
        elif len(self.shape_output) == 3:
            # Assert the shape correspond to (H, W, ?)
            assert self.shape_output[:2] == [
                H,
                W,
            ], f"Expected shape_output to start with {H, W} but got {self.shape_output[:2]}"
            # Apply a convolutional layer to get the right shape
            x = nn.Conv(
                features=self.shape_output[2],
                kernel_size=(1, 1),
                strides=(1, 1),
            )(x)

        # Apply the output activation function and return the output
        x = self.activation_output_fn(x)
        return x
        
