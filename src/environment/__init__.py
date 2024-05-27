from typing import Dict, Type
from src.environment.base_env import BaseEcoEnvironment
from src.environment.gridworld import GridworldEnv

env_name_to_EnvClass: Dict[str, Type[BaseEcoEnvironment]] = {
    "Gridworld": GridworldEnv,
}
