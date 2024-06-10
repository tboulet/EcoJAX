import hydra
from omegaconf import OmegaConf

def merge_container(*containers):
    """Merge containers of the same type (list, dict, tuple) into one container.
    """
    containers = [OmegaConf.to_container(container) for container in containers]
    if all(isinstance(container, list) for container in containers):
        return [item for container in containers for item in container]
    elif all(isinstance(container, dict) for container in containers):
        return {key: value for container in containers for key, value in container.items()}
    # elif all(isinstance(container, tuple) for container in containers):
    #     return tuple(item for container in containers for item in container)
    else:
        raise ValueError(f"All containers should be of the same type, but got {[type(container) for container in containers]}")


def register_hydra_resolvers():
    """Register the custom Hydra resolvers.
    """
    OmegaConf.register_new_resolver("merge", merge_container)
    OmegaConf.register_new_resolver("eval", eval) 