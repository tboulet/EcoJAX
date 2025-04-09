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
            if len(values) == 0:
                continue # skip empty histograms
            
            if np.min(values) == np.max(values):
                # Constant array: use 1 bin centered on the unique value
                single_val = values[0]
                bin_edges = np.array([single_val - 0.5, single_val + 0.5])
                counts = np.array([len(values)])
                histogram =  wandb.Histogram(np_histogram=(counts, bin_edges))
            else:
                # Non-constant array: use default histogram
                histogram = wandb.Histogram(values)
            self.run.log({key: histogram}, step=timestep)

    def log_maps(
        self,
        dict_maps: Dict[str, List[List[float]]],
        timestep: int,
    ):
        dict_return = {}
        for name, map in dict_maps.items():
            map = np.array(map)
            assert len(map.shape) == 2, f"map should be 2D, got {map.shape}"
            img_rgb = np.zeros((map.shape[0], map.shape[1], 3), dtype=np.float32)
            highest_abs_value = np.abs(map).max() 
            if highest_abs_value == 0:
                highest_abs_value = 1
            map /= highest_abs_value # map is in [-1, 1]. I want it red at -1 and blue at 1.
            img_rgb[..., 0] = (map < 0) * (255 * -map) # red channel
            img_rgb[..., 2] = (map > 0) * (255 * map) # blue channel
            dict_return[name] = wandb.Image(img_rgb, caption=name)
        wandb.log(dict_return, step=timestep)
            
    def close(self):
        self.run.finish()
