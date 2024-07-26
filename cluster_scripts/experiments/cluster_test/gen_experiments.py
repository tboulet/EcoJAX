#!/usr/bin/env python3
"""Script for generating experiment.txt"""
import itertools
import os
from datetime import datetime
from typing import Tuple

# define some paths
USER = os.environ["USER"]
PROJECT_HOME = os.path.dirname(os.path.abspath(USER))
EXPERIMENT_NAME = "cluster_test"
print(PROJECT_HOME)

def run_name(combo, keys):
    """Create a name for the run based on the parameter values"""
    combo_strings = "-".join(
        [
            f"{key.replace('.', '_')}_{value.lower() if isinstance(value, str) else value}"
            for key, value in zip(keys, combo)
        ]
    )
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    return f"{current_datetime}-{combo_strings}".rstrip("-")

# this is the base command that will be used for the experiment
base_output_dir = f"/disk/scratch/${USER}/projects/EcoJAX/outputs/{EXPERIMENT_NAME}"
base_call = f"python {PROJECT_HOME}/run.py log_dir={base_output_dir}"

# define a dictionary of variables to perform a grid search over.
# the key for each variable should match the name of the command-line
# argument required by the script in base_call
variables = {
    "env.p_base_plant_growth": [
        0.01,
        0.015,
        0.02,
    ]
}

combinations = list(itertools.product(*variables.values()))
print(f"Total experiments = {len(combinations)}")

output_file = open(
    f"{PROJECT_HOME}/cluster_scripts/experiments/{EXPERIMENT_NAME}/experiment.txt",
    "w+",
)

for c in combinations:
    rn = run_name(c, variables.keys())
    expt_call = f"{base_call} env.metrics.config_video.dir_videos={os.path.join(base_output_dir, rn)} +run_name={rn}"
    for i, var in enumerate(variables.keys()):
        expt_call += f" {var}={c[i]}"
    print(expt_call, file=output_file)

output_file.close()
