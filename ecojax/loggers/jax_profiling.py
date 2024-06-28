from ecojax.loggers import BaseLogger

from tensorboardX import SummaryWriter
from typing import Dict, List, Tuple, Type, Union
from jax import profiler


class LoggerJaxProfiling(BaseLogger):
    def __init__(
        self,
        log_dir: str = "logs/jax-trace",
        create_perfetto_link: bool = True,
        create_perfetto_trace: bool = False,
    ):
        self.pr = profiler.start_trace(
            log_dir=log_dir,
            create_perfetto_link=create_perfetto_link,
            create_perfetto_trace=create_perfetto_trace,
        )

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
        print("JAX profiling finished.")
