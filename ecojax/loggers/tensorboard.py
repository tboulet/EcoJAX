import numpy as np
from ecojax.loggers import BaseLogger

from tensorboardX import SummaryWriter
from typing import Dict, List, Tuple, Type, Union



class LoggerTensorboard(BaseLogger):
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)
    
    def log_scalars(self, dict_scalars: Dict[str, float], timestep: int):
        for name, value in dict_scalars.items():
            self.writer.add_scalar(name, value, timestep)
    
    def log_histograms(self, dict_histograms: Dict[str, List[float]], timestep: int):
        for name, values in dict_histograms.items():
            self.writer.add_histogram(name, values, timestep)

    def close(self):
        self.writer.close()
            