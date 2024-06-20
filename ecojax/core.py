from abc import ABC, abstractmethod
from flax import struct

import jax.numpy as jnp


@struct.dataclass
class EcoInformation:
    """This JAX data class represents the information about the dynamics of the environment."""

    # Whether agents have been born this timestep
    are_newborns_agents: jnp.ndarray

    # The indexe(s) of the parents of each newborn agent
    indexes_parents: jnp.ndarray

    # Whether agents have died this timestep
    are_just_dead_agents: jnp.ndarray
