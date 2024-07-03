from collections import defaultdict
import time
from typing import Any, Callable, Dict, Union


def timeit(func: Callable[..., Any]) -> Callable[..., Union[Any, float]]:
    """A wrapper function to return the result of the function and the time it took to execute it."""

    def time_measured_func(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time

    return time_measured_func


class RuntimeMeter:
    """A context manager class to measure the time of various stages of the code take.

    with RuntimeMeter("train") as rm:
        # Training code
        #  ...
    with RuntimeMeter("eval") as rm:
        # Evaluation code
        #  ...
    with RuntimeMeter("train") as rm:   # second training phase, but the RuntimeMeter will add the time to the first training phase
        # Training code
        #  ...

    training_time = RuntimeMeter.get_runtime("train")
    eval_time = RuntimeMeter.get_runtime("eval")
    """

    stage_name_to_runtime: Dict[str, float] = defaultdict(lambda: 0)
    stage_name_to_num_calls: Dict[str, int] = defaultdict(lambda: 0)

    @staticmethod
    def get_stage_runtime(stage_name: str) -> float:
        """Return the cumulative time taken by the stage.
        If the stage_name is "total", it will return the total time taken by all stages.
        If the stage_name is not found, it will return 0.

        Args:
            stage_name (str): the name of the stage, as it was used in the context manager.

        Returns:
            float: the cumulative time taken by the stage.
        """
        if stage_name == "total":
            return sum(RuntimeMeter.stage_name_to_runtime.values())
        elif stage_name not in RuntimeMeter.stage_name_to_runtime:
            return 0
        else:
            return RuntimeMeter.stage_name_to_runtime[stage_name]

    @staticmethod
    def get_averaged_stage_runtime(stage_name: str) -> float:
        """Return the average time taken by the stage.

        Args:
            stage_name (str): the name of the stage, as it was used in the context manager.
                        
        Returns:
            float: the average time taken by the stage.
        """
        if stage_name not in RuntimeMeter.stage_name_to_runtime:
            return 0
        return (
            RuntimeMeter.stage_name_to_runtime[stage_name]
            / RuntimeMeter.stage_name_to_num_calls[stage_name]
        )

    @staticmethod
    def get_runtimes() -> Dict[str, float]:
        """Return a dictionnary mapping the stage names to the cumulative time taken by the stage.

        Returns:
            Dict[str, float]: the dictionnary mapping the stage names to the cumulative time taken by the stage.
        """
        return dict(RuntimeMeter.stage_name_to_runtime)

    @staticmethod
    def get_average_runtimes() -> Dict[str, float]:
        """Return a dictionnary mapping the stage names to the average time taken by the stage.

        Returns:
            Dict[str, float]: the dictionnary mapping the stage names to the average time taken by the stage.
        """
        return {
            stage_name: RuntimeMeter.get_averaged_stage_runtime(stage_name)
            for stage_name in RuntimeMeter.stage_name_to_runtime
        }
        
    @staticmethod
    def get_total_runtime() -> float:
        """Return the total time taken by all stages.

        Returns:
            float: the total time taken by all stages.
        """
        return sum(RuntimeMeter.stage_name_to_runtime.values())

    def __init__(self, stage_name: str, n_calls: int = 1):
        """Initialize the RuntimeMeter.

        Args:
            stage_name (str): a string identifying the stage.
            n_calls (int, optional): the number of calls to the stage. Defaults to 1.
        """
        self.stage_name = stage_name
        self.n_calls = n_calls

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stage_name_to_runtime[self.stage_name] += time.time() - self.start_time
        self.stage_name_to_num_calls[self.stage_name] += self.n_calls


def get_runtime_metrics():
    """Return the metrics of the runtimes.

    Returns:
        Dict[str, float]: a dictionnary mapping the stage names to the cumulative and averaged time taken by the stage.
    """
    dict_runtime_metrics = {}
    for stage_name in RuntimeMeter.stage_name_to_runtime:
        dict_runtime_metrics[f"runtime/{stage_name}"] = RuntimeMeter.get_stage_runtime(stage_name)
        dict_runtime_metrics[f"runtime/{stage_name}_avg"] = RuntimeMeter.get_averaged_stage_runtime(stage_name)
    return dict_runtime_metrics
    
if __name__ == "__main__":
    import time
    import random

    def foo():
        time.sleep(0.1)

    def bar():
        time.sleep(0.2)

    for _ in range(3):
        with RuntimeMeter("foo"):
            foo()
        with RuntimeMeter("bar"):
            bar()

    print(RuntimeMeter.get_stage_runtime("foo"))
    print(RuntimeMeter.get_stage_runtime("bar"))
    print(RuntimeMeter.get_stage_runtime("total"))
