from ecojax.loggers import BaseLogger

from typing import Dict, List, Tuple, Type, Union
from jax import profiler


class LoggerJaxProfiling(BaseLogger):
    def __init__(
        self,
        log_dir: str = "./tensorboard",
    ):
        profiler.start_trace(log_dir)
        
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
        profiler.stop_trace()
        print("[Logging] JAX Profiling finished. See results on tensorboard with command `tensorboard --logdir tensorboard` in section #profile.")
