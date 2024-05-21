from typing import Dict, Type
from src.environment.base_env import BaseEnvironment
from src.environment.gridworld import GridworldEnv

env_name_to_EnvClass : Dict[str, Type[GridworldEnv]] = {  # temp typing
    "Gridworld" : GridworldEnv,
}