import numpy as np
from ecojax.loggers import BaseLogger

from tensorboardX import SummaryWriter
from typing import Dict, List, Tuple, Type, Union


class LoggerTensorboard(BaseLogger):
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalars(
        self,
        dict_scalars: Dict[str, float],
        timestep: int,
    ):
        for name, value in dict_scalars.items():
            self.writer.add_scalar(name, value, timestep)

    def log_histograms(
        self,
        dict_histograms: Dict[str, List[float]],
        timestep: int,
    ):
        for name, values in dict_histograms.items():
            values = values[~np.isnan(values)]
            if len(values) > 0:
                try:
                    self.writer.add_histogram(name, values, timestep)
                except:
                    print(f"Error in logging histogram {name}, values: {values}")
    
    def log_maps(
        self,
        dict_maps: Dict[str, List[List[float]]],
        timestep: int,
    ):
        for name, map in dict_maps.items():
            map = np.array(map)
            image = np.zeros((3, map.shape[0], map.shape[1]))
            highest_value = map.max()
            if highest_value > 0:  
                image[2, map > 0] = map[map > 0] / highest_value
            lowest_value = map.min()
            if lowest_value < 0:
                image[0, map < 0] = -map[map < 0] / -lowest_value
            self.writer.add_image(name, image, timestep)
        
    def close(self):
        self.writer.close()
