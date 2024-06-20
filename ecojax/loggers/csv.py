import csv
import os

import jax
import numpy as np
from ecojax.core import EcoInformation
from ecojax.evolution.metrics import compute_eco_return, get_phylogenetic_tree
from ecojax.loggers import BaseLogger

from tensorboardX import SummaryWriter
from typing import Dict, List, Tuple, Type, Union


class LoggerCSV(BaseLogger):
    def __init__(
        self,
        dir_metrics: str,
        do_log_phylo_tree: bool = True,
        period_compute_metrics: int = 5000,
    ):
        # Metrics
        os.makedirs(os.path.dirname(dir_metrics), exist_ok=True)
        self.file_csv_metrics = open(
            f"{dir_metrics}/metrics.csv", "w", newline="", encoding="utf-8"
        )
        self.writer_metrics_csv = csv.writer(self.file_csv_metrics)
        self.writer_metrics_csv.writerow(
            ["timestep", "metric_name", "agent_idx", "value"]
        )
        # Eco return metrics
        os.makedirs(os.path.dirname(dir_metrics), exist_ok=True)
        self.path_eco_return_metrics = f"{dir_metrics}/eco_return_metrics.csv"
        self.period_compute_metrics = period_compute_metrics
        self.current_agent_idx_to_id = {}  # map current agent index to its ID
        self.id_to_agent_idx = {}  # map ID to the agent index when he was living
        self.id_to_timestep_born = {}  # map ID to the timestep when he was born
        self.id_to_parent_id = {}  # map ID to the parent ID
        # Phylo tree
        self.do_log_phylo_tree = do_log_phylo_tree
        if self.do_log_phylo_tree:
            self.path_phylo_tree = f"{dir_metrics}/phylo_tree.png"

    def log_scalars(
        self,
        dict_scalars: Dict[str, float],
        timestep: int,
    ):
        for metric_name, scalar in dict_scalars.items():
            self.writer_metrics_csv.writerow(
                [
                    timestep,
                    metric_name,
                    "",
                    float(scalar),
                ]
            )

    def log_histograms(
        self,
        dict_histograms: Dict[str, List[float]],
        timestep: int,
    ):
        for metric_name, histogram in dict_histograms.items():
            for agent_idx in range(len(histogram)):
                value = histogram[agent_idx]
                if not np.isnan(value):
                    self.writer_metrics_csv.writerow(
                        [
                            timestep,
                            metric_name,
                            agent_idx,
                            value,
                        ]
                    )

    def log_eco_metrics(
        self,
        eco_information: EcoInformation,
        timestep: int,
    ):
        return  # Not implemented for eco info
        # Deal with deaths : we release the current agent indexes
        for agent_idx in list_deaths:
            self.current_agent_idx_to_id.pop(agent_idx)

        # Deal with reproduction
        for agent_idx, list_parents_idx in dict_reproduction.items():
            assert (agent_idx not in self.current_agent_idx_to_id) or (
                agent_idx in list_deaths
            ), f"Agent is already a current agent and is not dead either: {agent_idx}"
            # If the agent has no parent, it means it spontaneously appeared. We add it to the phylo tree.
            if len(list_parents_idx) == 0 or list_parents_idx[0] == -1:
                id_agent = len(self.id_to_agent_idx)  # generate a new ID
                self.current_agent_idx_to_id[agent_idx] = id_agent
                self.id_to_agent_idx[id_agent] = agent_idx
                self.id_to_timestep_born[id_agent] = timestep
                # We add -1 as parent ID
                self.id_to_parent_id[id_agent] = -1
            # If the agent has a parent, we do the same
            else:
                id_agent = len(self.id_to_agent_idx)  # generate a new ID
                self.current_agent_idx_to_id[agent_idx] = id_agent
                self.id_to_agent_idx[id_agent] = agent_idx
                self.id_to_timestep_born[id_agent] = timestep
                # We also link the agent to its parent
                parent_idx = list_parents_idx[0]
                assert (
                    parent_idx in self.current_agent_idx_to_id
                ), f"Parent index not found in current agents: {parent_idx}"
                id_parent = self.current_agent_idx_to_id[parent_idx]
                assert (
                    id_parent in self.id_to_agent_idx
                ), f"Parent ID not found: {id_parent}"
                self.id_to_parent_id[id_agent] = id_parent

        # Periodically log the phylo tree
        if timestep % self.period_compute_metrics == 0:
            # Compute the eco return
            self.id_to_eco_return = compute_eco_return(
                id_to_parent_id=self.id_to_parent_id,
                discount_factor=0.9,
            )

            # Log the eco return metric
            self.file_csv_phylo_tree = open(
                self.path_eco_return_metrics, "w", newline="", encoding="utf-8"
            )
            writer_phylo_tree = csv.writer(self.file_csv_phylo_tree)
            writer_phylo_tree.writerow(
                ["timestep", "metric_name", "agent_idx", "value"]
            )

            for id_agent, eco_return in self.id_to_eco_return.items():
                agent_idx = self.id_to_agent_idx[id_agent]
                time_born = self.id_to_timestep_born[id_agent]
                writer_phylo_tree.writerow(
                    [
                        time_born,
                        "eco_return",
                        agent_idx,
                        eco_return,
                    ]
                )
                
            # Save the phylo tree
            if self.do_log_phylo_tree:
                phylotree_fig = get_phylogenetic_tree(
                    id_to_parent_id=self.id_to_parent_id,
                    id_to_timestep_born=self.id_to_timestep_born,
                )
                phylotree_fig.savefig(self.path_phylo_tree)
                print(f"Phylo tree saved at {self.path_phylo_tree}")

    def close(self):
        try:
            self.file_csv_metrics.close()
            if self.do_log_phylo_tree:
                self.file_csv_phylo_tree.close()
        except Exception as e:
            print(f"Error while closing the logger: {e}")