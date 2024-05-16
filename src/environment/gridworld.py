# Gridworld EcoJAX environment

from typing import Any
import jax.numpy as jnp

from src.environment.base_env import BaseEnvironment


class GridworldEnv(BaseEnvironment):
    """The Gridworld environment.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.width = config["width"]
        self.height = config["height"]

    def start(self):
        self.map = jnp.zeros(shape=(self.width, self.height, 3))

    def step(self, actions: jnp.ndarray = None):
        pass

    def get_RGB_map(self) -> Any:
        return self.map
