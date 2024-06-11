import cProfile
from ecojax.loggers import BaseLogger

from tensorboardX import SummaryWriter
from typing import Dict, List, Tuple, Type, Union



class LoggerSnakeviz(BaseLogger):
    def __init__(self):
        self.pr = cProfile.Profile()
        self.pr.enable()
    
    def log_scalars(self, dict_scalars: Dict[str, float], timestep: int):
        pass
    
    def log_histograms(self, dict_histograms: Dict[str, List[float]], timestep: int):
        pass

    def close(self):
        self.pr.disable()
        self.pr.dump_stats("logs/profile_stats.prof")
        print("Profile stats dumped to logs/profile_stats.prof")
            