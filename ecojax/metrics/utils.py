from typing import Any, Dict, Tuple
from ecojax.utils import is_array, is_scalar

import jax.numpy as jnp
import numpy as np


def get_dict_metrics_by_type(metrics: Dict[str, Any]) -> Tuple[Dict[str, Any]]:
    metrics_scalar = {}
    metrics_histogram = {}
    metrics_map = {}
    for key, value in metrics.items():
        try:
            value = np.array(value)
        except Exception as e:
            raise f"Error converting metric {key} of value's type {type(value)} to numpy array: {e}"
        if is_scalar(value):
            metrics_scalar[key] = value
        elif isinstance(value, np.ndarray):
            if value.ndim == 1:
                metrics_histogram[key] = value
            elif value.ndim == 2:
                metrics_map[key] = value
            else:
                raise ValueError(f"Invalid shape for metric {key}: {value.shape}")
        else:
            raise ValueError(f"Invalid metric type: {type(value)}")
    return metrics_scalar, metrics_histogram, metrics_map
