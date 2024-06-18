from typing import Any, Dict
from ecojax.utils import is_array, is_scalar


def get_dicts_metrics(metrics : Dict[str, Any]):
    metrics_scalar = {}
    metrics_histogram = {}
    for key, value in metrics.items():
        if is_scalar(value):
            metrics_scalar[key] = value
        elif is_array(value):
            metrics_histogram[key] = value
        else:
            raise ValueError(f"Invalid metric type: {type(value)}")
    return metrics_scalar, metrics_histogram