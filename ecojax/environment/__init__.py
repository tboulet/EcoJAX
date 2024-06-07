from typing import Dict, Type
from ecojax.environment.base_env import BaseEcoEnvironment
from ecojax.environment.gridworld import GridworldEnv

env_name_to_EnvClass: Dict[str, Type[BaseEcoEnvironment]] = {
    "Gridworld": GridworldEnv,
}
