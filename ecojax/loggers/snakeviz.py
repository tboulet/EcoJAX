import cProfile
from ecojax.loggers import BaseLogger

from tensorboardX import SummaryWriter
from typing import Dict, List, Tuple, Type, Union


class LoggerSnakeviz(BaseLogger):
    def __init__(self):
        self.pr = cProfile.Profile()
        self.pr.enable()

    def log_scalars(
        self,
        *args,
        **kwargs,
    ) -> None:
        pass

    def log_histograms(
        self,
        *args,
        **kwargs,
    ) -> None:
        pass

    def close(self):
        self.pr.disable()
        self.pr.dump_stats("logs/profile_stats.prof")
        print("[Profile] Profile stats dumped to logs/profile_stats.prof")
