import csv
import os
from ecojax.loggers import BaseLogger

from tensorboardX import SummaryWriter
from typing import Dict, List, Tuple, Type, Union


class LoggerCSV(BaseLogger):
    def __init__(self, path_csv: str):
        os.makedirs(os.path.dirname(path_csv), exist_ok=True)
        self.file_csv = open(path_csv, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.file_csv)

    def log_scalars(self, dict_scalars: Dict[str, float], timestep: int):
        for metric_name, scalar in dict_scalars.items():
            self.csv_writer.writerow(
                [
                    timestep,
                    metric_name,
                    "",
                    float(scalar),
                ]
            )

    def log_histograms(self, dict_histograms: Dict[str, List[float]], timestep: int):
        for metric_name, histogram in dict_histograms.items():
            for agent_id in range(len(histogram)):
                self.csv_writer.writerow(
                    [
                        timestep,
                        metric_name,
                        agent_id,
                        float(histogram[agent_id]),
                    ]
                )

    def close(self):
        self.file_csv.close()
