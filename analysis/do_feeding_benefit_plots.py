from collections import defaultdict
import itertools
import os
import sys
from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from analysis.utils import (
    load_metrics_data,
    load_event_data,
    compute_feeding_stats,
    compute_times_fed_vs_life_exp,
    do_line_plot,
)
from analysis.constants import (
    xlabel_dict,
    title_dict
)

plot_kwargs = {"linewidth": 5, "color": "#EE1195"}

data_dir, load = os.path.join("outputs/feeding"), False
if len(sys.argv) > 1:
    load = bool(int(sys.argv[1]))

var_names = os.listdir(data_dir)
for var_name in var_names:
    expt_dir = os.path.join(data_dir, var_name)
    if not os.path.isdir(expt_dir):
        continue

    results_fp = os.path.join(expt_dir, "overall_results.csv")
    if load and os.path.exists(results_fp):
        results = pd.read_csv(results_fp)
    else:
        results = []
        for run_name in tqdm(os.listdir(expt_dir), desc=expt_dir):
            if not run_name.startswith("2024"):
                continue
            try:
                metrics_data, _ = load_metrics_data(expt_dir, run_name)
                if metrics_data is None:
                    continue

                feeding_data = load_event_data(expt_dir, run_name, "feed")
                death_data = load_event_data(expt_dir, run_name, "death")
                feeding_data, feeder_counts, feedee_counts, _ = (
                    compute_feeding_stats(feeding_data, death_data, age_cutoff=100)
                )

                # compute average number of agents over last 50k timesteps
                n_agents = metrics_data["n_agents"].mean()

                # compute average effect of being fed on life expectancy
                le_data = compute_times_fed_vs_life_exp(feedee_counts, death_data)
                feeding_benefit = np.nan
                if le_data:
                    baseline, tmp = le_data[0], []
                    for k, v in le_data.items():
                        if k == 0:
                            continue
                        tmp.append(v - baseline)
                    feeding_benefit = 100 * np.mean(tmp) / np.maximum(baseline, 1)

                results.append(
                    {
                        var_name: metrics_data[var_name].values[0],
                        "seed": metrics_data["seed"].values[0],
                        "n_agents": n_agents,
                        "feeding_benefit": feeding_benefit,
                    }
                )
            except Exception as e:
                continue

    results = pd.DataFrame(results)
    results.to_csv(results_fp, index=False)

    # plot feeding benefit against variable
    results = results[results["feeding_benefit"].isna() == False]
    fig, ax = plt.subplots(figsize=(20, 10))
    do_line_plot(
        ax,
        results,
        var_name,
        "feeding_benefit",
        plot_kwargs=plot_kwargs
    )
    ax.set(
        title=f"Effect of {title_dict[var_name]} on benefit of being fed",
        xlabel=xlabel_dict[var_name],
        ylabel="Avg % increase in life exp. from being fed"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(data_dir, f"feeding_benefit_{var_name}.svg"))
    plt.close(fig)
