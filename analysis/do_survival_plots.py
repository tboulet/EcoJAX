from collections import defaultdict
import itertools
import os
import time
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from analysis.utils import (
    load_metrics_data,
    do_line_plot,
)
from analysis.constants import (
    xlabel_dict,
    title_dict
)

plot_kwargs = {"linewidth": 5, "color": "#EE1195"}

data_dir, load, dfs = os.path.join("outputs/no_feeding"), False, []
if len(sys.argv) > 1:
    load = bool(int(sys.argv[1]))

last_n_steps = 1e4

var_names = os.listdir(data_dir)
for var_name in var_names:
    expt_dir = os.path.join(data_dir, var_name)
    if not os.path.isdir(expt_dir):
        continue

    summary_data_fp = os.path.join(expt_dir, "summary_data.csv")
    if load and os.path.exists(summary_data_fp):
        summary_data = pd.read_csv(summary_data_fp)
    else:
        summary_data = []
        for run_name in tqdm(os.listdir(expt_dir)):
            if not run_name.startswith("2024"):
                continue
            try:
                metrics_data, _ = load_metrics_data(data_dir, run_name, last_n_steps=last_n_steps)

                # compute average metric values over last n timesteps
                life_exp = metrics_data["life_expectancy"].mean()
                n_agents = metrics_data["n_agents"].mean()
                survival_to_adulthood = metrics_data["survival_to_adulthood"].mean()

                summary_data.append(
                    {
                        var_name: metrics_data[var_name].values[0],
                        "seed": metrics_data["seed"].values[0],
                        "life_expectancy": life_exp,
                        "n_agents": n_agents,
                        "survival_to_adulthood": survival_to_adulthood,
                    }
                )
            except Exception as e:
                tqdm.write(f"Error processing run {run_name}: {e}")
                continue


        summary_data = pd.DataFrame(summary_data)
        summary_data.fillna(0, inplace=True)
        summary_data.to_csv(os.path.join(expt_dir, "summary_data.csv"), index=False)

    fig, ax = plt.subplots(1, 1, figsize=(20, 10), sharex=True)
    do_line_plot(
        ax,
        summary_data,
        var_name,
        "survival_to_adulthood",
        plot_kwargs=plot_kwargs,
    )
    ax.set(
        title=f"Infant survival rate vs {title_dict[var_name]}",
        xlabel=xlabel_dict[var_name],
        ylabel="Infant survival rate",
    )

    fig.tight_layout()
    for fmt in ["png", "svg"]:
        fig.savefig(os.path.join(data_dir, f"infant_survival_{var_name}.{fmt}"))
    plt.close(fig)
