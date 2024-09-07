import importlib
from typing import Any, Dict, List, Union
from abc import ABC, abstractmethod
from functools import partial

import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from flax import struct
import flax.linen as nn
from jax.lib import xla_bridge

def to_numeric(x: Union[int, float, str, None]) -> Union[int, float]:
    if isinstance(x, int) or isinstance(x, float):
        return x
    elif isinstance(x, str):
        return float(x)
    elif x == "inf":
        return float("inf")
    elif x == "-inf":
        return float("-inf")
    elif x is None:
        return None
    else:
        raise ValueError(f"Cannot convert {x} to numeric")


def is_scalar(data):
    """
    Detect if the given data is a scalar.

    Parameters:
    data : any type
        The data to be checked.

    Returns:
    bool
        True if data is a scalar, False otherwise.
    """
    if isinstance(data, (int, float, bool)):
        return True
    elif np.isscalar(data):
        return True
    elif isinstance(data, np.ndarray) and data.shape == ():  # numpy array with a single element
        return True
    else:
        return False
    

def is_array(data):
    """
    Detect if the given data is an array.

    Parameters:
    data : any type
        The data to be checked.

    Returns:
    bool
        True if data is an array, False otherwise.
    """
    return isinstance(data, np.ndarray)
    
def try_get_seed(config: Dict) -> int:
    """Will try to extract the seed from the config, or return a random one if not found

    Args:
        config (Dict): the run config

    Returns:
        int: the seed
    """
    try:
        seed = config["seed"]
        if not isinstance(seed, int):
            seed = np.random.randint(0, 1000)
    except KeyError:
        seed = np.random.randint(0, 1000)
    return seed


def try_get(
    dictionnary: Dict, key: str, default: Union[int, float, str, None] = None
) -> Any:
    """Will try to extract the key from the dictionary, or return the default value if not found
    or if the value is None

    Args:
        x (Dict): the dictionary
        key (str): the key to extract
        default (Union[int, float, str, None]): the default value

    Returns:
        Any: the value of the key if found, or the default value if not found
    """
    try:
        return dictionnary[key] if dictionnary[key] is not None else default
    except KeyError:
        return default


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


def logit(x, eps=1e-8):
    x = jnp.clip(x, eps, 1 - eps)
    return jnp.log(x / (1 - x))


DICT_COLOR_TAG_TO_RGB = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "cyan": (0.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
    "black": (0.0, 0.0, 0.0),
    "white": (1.0, 1.0, 1.0),
    "gray": (0.5, 0.5, 0.5),
    "orange": (1.0, 0.5, 0.0),
    "purple": (0.5, 0.0, 0.5),
    "pink": (1.0, 0.5, 0.5),
    "brown": (0.6, 0.3, 0.0),
    "lime": (0.5, 1.0, 0.0),
}


def nest_for_array(func):
    """Decorator to allow a function to be applied to nested arrays.

    Args:
        func (function): the function to decorate

    Returns:
        function: the decorated function
    """

    def wrapper(arr, *args, **kwargs):
        if isinstance(arr, jnp.ndarray):
            return func(arr, *args, **kwargs)
        elif isinstance(arr, dict):
            if "key_random" in kwargs:
                key_random = kwargs["key_random"]
                del kwargs["key_random"]
                for key, value in arr.items():
                    key_random, subkey = random.split(key_random)
                    arr[key] = wrapper(value, *args, key_random=subkey, **kwargs)
            else:
                for key, value in arr.items():
                    arr[key] = wrapper(value, *args, **kwargs)
            return arr
        elif isinstance(arr, list):
            if "key_random" in kwargs:
                key_random = kwargs["key_random"]
                del kwargs["key_random"]
                for idx, value in enumerate(arr):
                    key_random, subkey = random.split(key_random)
                    arr[idx] = wrapper(value, *args, key_random=subkey, **kwargs)
            else:
                for idx, value in enumerate(arr):
                    arr[idx] = wrapper(value, *args, **kwargs)
            return arr
        else:
            raise ValueError(f"Unknown type for array: {type(arr)}")

    return wrapper


def get_dict_flattened(d, parent_key='', sep='.'):
    """Get a flattened version of a nested dictionary, where keys correspond to the path to the value.

    Args:
        d (Dict): The dictionary to be flattened.
        parent_key (str): The base key string (used in recursive calls).
        sep (str): Separator to use between keys.

    Returns:
        Dict: The flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(get_dict_flattened(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

    
def instantiate_class(**kwargs) -> Any:
    """Instantiate a class from a dictionnary that contains a key "class_string" with the format "path.to.module:ClassName"
    and that contains other keys that will be passed as arguments to the class constructor

    Args:
        config (dict): the configuration dictionnary
        **kwargs: additional arguments to pass to the class constructor

    Returns:
        Any: the instantiated class
    """
    assert (
        "class_string" in kwargs
    ), "The class_string should be specified in the config"
    class_string: str = kwargs["class_string"]
    module_name, class_name = class_string.split(":")
    module = importlib.import_module(module_name)
    Class = getattr(module, class_name)
    object_config = kwargs.copy()
    object_config.pop("class_string")
    return Class(**object_config)



def check_jax_device():
    try:
        print("Checking device used by JAX:")
        print(f"\tAvailable devices: {jax.devices()}")
        print(f"\tPlatform: {xla_bridge.get_backend().platform}")
    except Exception as e:
        print(f"Error while checking JAX device: {e}")
        

def jprint(x, msg = None):
    """Print the value of x using JAX's print function, even inside of a JAX jit function"""
    if msg is not None:
        jax.debug.print(msg + " :")
        jax.debug.print("{x}", x=x)
    else:
        jax.debug.print("{x}", x=x)
    jax.debug.print("")
        
def jbreakpoint():
    """Breakpoint inside a JAX jit function"""
    jax.debug.breakpoint()
    
def jprint_and_breakpoint(x):
    """Print the value of x using JAX's print function, even inside of a JAX jit function"""
    jax.debug.print("{x}", x=x)
    jax.debug.breakpoint()
    

def add_scalars_as_channels_single(image: jnp.ndarray, scalars: jnp.ndarray) -> jnp.ndarray:
    """
    Concatenates scalar observations as additional channels to a single visual field (image).
    
    Args:
        image (jnp.ndarray): A visual observation tensor of shape (H, W, C).
        scalars (jnp.ndarray): A scalar observation tensor of shape (num_scalars).
    
    Returns:
        jnp.ndarray: The new image tensor with scalar observations as additional channels.
    """
    H, W, C = image.shape
    num_scalars = scalars.shape[0]

    # Reshape scalars to (1, 1, num_scalars), so they can be broadcasted across height and width
    scalars_expanded = jnp.reshape(scalars, (1, 1, num_scalars))

    # Broadcast scalars to the image shape (H, W, num_scalars)
    scalars_broadcasted = jnp.broadcast_to(scalars_expanded, (H, W, num_scalars))

    # Concatenate along the channel axis (axis=-1)
    image_with_scalars = jnp.concatenate([image, scalars_broadcasted], axis=-1)

    return image_with_scalars


def average_pooling(input_array: jnp.ndarray, h: int) -> jnp.ndarray:
    """
    Perform average pooling on an array of shape (3h, 3h, C) to reduce it to (3, 3, C).
    
    Args:
        input_array (jnp.ndarray): Input array of shape (3h, 3h, C).
        h (int): The height and width of each pooling window.
    
    Returns:
        jnp.ndarray: Pooled array of shape (3, 3, C).
    """
    # Get the shape of the input array
    H, W, C = input_array.shape
    assert H == 3 * h and W == 3 * h, "Input dimensions must be (3h, 3h, C)"
    
    # Reshape the array to group (h, h) blocks
    reshaped = input_array.reshape(3, h, 3, h, C)
    
    # Take the mean over the height and width of each (h, h) block
    pooled = reshaped.mean(axis=(1, 3))
    
    return pooled


def separate_visual_field(input_array: jnp.ndarray) -> jnp.ndarray:
    """
    Separates the visual field into 5 regions (center, front, left, right, and backward),
    with backward excluding diagonal tiles, but front including them.
    
    Args:
        input_array (jnp.ndarray): Input array of shape (H, H, C), where H is the height/width and C is the number of channels.
    
    Returns:
        jnp.ndarray: Array of shape (5, C) where each row corresponds to the averaged values in each region.
    """
    H, W, C = input_array.shape
    assert H == W, "Input array must have square spatial dimensions (H, H, C)."
    
    # Center pixel (middle of the array)
    center_pixel = input_array[H // 2, W // 2, :]

    # Create a meshgrid of indices to define regions
    y, x = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij')
    
    # Define masks for each region
    center_mask = (y == H // 2) & (x == W // 2)

    # Front: Upper half of the image bounded by two diagonals (includes diagonal)
    front_mask = (y < H // 2) & (x >= y) & (x < H - y)

    # Left: Left of the center pixel (includes diagonal in lower half)
    left_mask = (x < H // 2) & ((y > x) | (y > H - x - 1))

    # Right: Right of the center pixel (includes diagonal in lower half)
    right_mask = (x > H // 2) & ((y >= H - x) | (y > x))

    # Backward: Lower half of the image, excluding diagonals
    backward_mask = (y > H // 2) & (x > H - y - 1) & (x < y)

    # Average the values in each region
    center_avg = jnp.sum(input_array * center_mask[..., None] / center_mask.sum(), axis=(0, 1))
    front_avg = jnp.sum(input_array * front_mask[..., None] / front_mask.sum(), axis=(0, 1))
    left_avg = jnp.sum(input_array * left_mask[..., None] / left_mask.sum(), axis=(0, 1))
    right_avg = jnp.sum(input_array * right_mask[..., None] / right_mask.sum(), axis=(0, 1))
    backward_avg = jnp.sum(input_array * backward_mask[..., None] / backward_mask.sum(), axis=(0, 1))
    
    # Stack the results into an array of shape (5, C)
    regions_avg = jnp.stack([center_avg, front_avg, left_avg, right_avg, backward_avg], axis=0)
    
    return regions_avg