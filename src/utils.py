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


def logit(x):
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