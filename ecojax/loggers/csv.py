import csv
import os

import jax
import numpy as np
import yaml
from ecojax.core.eco_info import EcoInformation
from ecojax.evolution.metrics import compute_eco_return, get_phylogenetic_tree
from ecojax.loggers import BaseLogger

from tensorboardX import SummaryWriter
from typing import Dict, List, Tuple, Type, Union


class LoggerCSV(BaseLogger):
    def __init__(
        self,
        log_dir: str,
        config_run: Dict,
        timestep_key: str = "_step",
    ):
        os.makedirs(log_dir, exist_ok=True)
        self.timestep_key = timestep_key
        # Log config as yaml
        with open(f"{log_dir}/config.yaml", "w") as f:
            yaml.dump(config_run, f)
        # Initialize scalar logger

        self.csv_path = os.path.join(log_dir, "scalars.csv")
        self.headers = [self.timestep_key]
        self.seen_fields = set(self.headers)

        # Create empty file and write initial header
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writeheader()

    def log_scalars(
        self,
        dict_scalars: Dict[str, float],
        timestep: int,
    ):
        row_dict = {self.timestep_key: timestep, **dict_scalars}
        new_keys = [k for k in dict_scalars if k not in self.seen_fields]

        if new_keys:
            # Update header and seen fields
            self.headers.extend(new_keys)
            self.seen_fields.update(new_keys)
            self._expand_csv_with_new_keys(new_keys)

        # Build full row with missing values as ""
        complete_row = {key: row_dict.get(key, "") for key in self.headers}

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow(complete_row)

    def _expand_csv_with_new_keys(self, new_keys):
        # Read all rows
        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)

        # Update rows with NaN ("") for new keys
        updated_rows = []
        for row in existing_rows:
            for key in new_keys:
                row[key] = ""
            updated_rows.append(row)

        # Rewrite file with updated headers and rows
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writeheader()
            writer.writerows(updated_rows)


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
        pass
