import numpy as np
from tqdm import tqdm
from ecojax.loggers import BaseLogger

from tensorboardX import SummaryWriter
from typing import Dict, List, Tuple, Type, Union


class LoggerTQDM(BaseLogger):
    def __init__(self, n_timesteps: int):
        self.progress_bar = tqdm(total=n_timesteps)
        self.last_timestep = 0

    def log_scalars(
        self,
        dict_scalars: Dict[str, float],
        timestep: int,
    ):
        dt = timestep - self.last_timestep
        if dt > 0:
            self.progress_bar.update(dt)
            self.last_timestep = timestep

    def log_histograms(
        self,
        *args,
        **kwargs,
    ) -> None:
        pass

    def close(self):
        self.progress_bar.close()
