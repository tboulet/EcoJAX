import numpy as np
import wandb
from ecojax.loggers import BaseLogger

from tensorboardX import SummaryWriter
from typing import Dict, List, Tuple, Type, Union


class LoggerWandB(BaseLogger):
    def __init__(self, name_run: str, config_run: Dict, **kwargs):
        self.run = wandb.init(
            name=name_run,
            config=config_run,
            **kwargs,
        )

    def log_scalars(
        self,
        dict_scalars: Dict[str, float],
        timestep: int,
    ):
        self.run.log(dict_scalars, step=timestep)

    def log_histograms(
        self,
        dict_histograms: Dict[str, List[float]],
        timestep: int,
    ):
        for key, values in dict_histograms.items():
            values = values[~np.isnan(values)]
            self.run.log({key: wandb.Histogram(values)}, step=timestep)

    def close(self):
        self.run.finish()
