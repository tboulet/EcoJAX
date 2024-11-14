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
    load_eval_data,
    compute_feeding_stats,
    compute_times_fed_vs_life_exp,
    do_regression,
    do_line_plot,
    do_regression_plot
)

scatter_plot_kwargs = {"s": 60, "c": "#46327E", "alpha": 0.7}
regression_plot_kwargs = {"linewidth": 4, "color": "#00BB83"}

data_dir, load, dfs = os.path.join("outputs/control"), False, []
if len(sys.argv) > 1:
    load = bool(int(sys.argv[1]))

def plot_regressions(df, var_name):
    print(f"\n\nVAR_NAME: {var_name}")
    # do regression plots for feeding amount and selectivity against feeding benefit
    fig, axs = plt.subplots(2, 1, figsize=(9, 16), sharex=True)
    for i, y_var in enumerate(["log_amount_feeding", "log_feeding_selectivity"]):
        reg_results = do_regression(df, "feeding_benefit", y_var, add_const=True)
        do_regression_plot(
            axs[i],
            df,
            "feeding_benefit",
            y_var,
            reg_results,
            scatter_kwargs=scatter_plot_kwargs,
            line_kwargs=regression_plot_kwargs,
        )
    axs[0].set(title="", xlabel="", ylabel="log(1 + feeding amount)")
    axs[1].set(title="", xlabel="Feeding benefit", ylabel="log(1 + feeding selectivity)")
    fig.tight_layout()
    for fmt in ["svg"]:
        fig.savefig(os.path.join(data_dir, f"benefit_regressions_{var_name}.{fmt}"))

    # do regression plot for feeding amount against feeding selectivity
    fig, ax = plt.subplots(figsize=(8, 10))
    reg_results = do_regression(
        df,
        "feeding_selectivity",
        "amount_feeding",
        add_const=False
    )
    do_regression_plot(
        ax,
        df,
        "feeding_selectivity",
        "amount_feeding",
        reg_results,
        scatter_kwargs=scatter_plot_kwargs,
        line_kwargs=regression_plot_kwargs,
    )
    ax.set(title="", xlabel="Feeding selectivity", ylabel="Feeding amount")
    fig.tight_layout()
    for fmt in ["svg"]:
        fig.savefig(
            os.path.join(data_dir, f"amount_selectivity_regression_{var_name}.{fmt}")
        )

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
                metrics_data, _ = load_metrics_data(expt_dir, run_name, last_n_steps=2.5 * 1e4)
                if metrics_data is None:
                    continue

                feeding_data = load_event_data(expt_dir, run_name, "feed")
                death_data = load_event_data(expt_dir, run_name, "death")
                feeding_data, feeder_counts, feedee_counts, avg_age_diff = (
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
                        if k > 0:
                            tmp.append(v - baseline)
                    feeding_benefit = 100 * np.mean(tmp) / np.maximum(baseline, 1)

                # compute average amount of feeding over last 100k timesteps
                # and normalise by the populations size
                amount_feeding = len(feeding_data) / max(n_agents, 1)

                # and avereage selectivity of feeding towards offspring
                selectivity = metrics_data["excess_offspring_feeding"].mean()
                selectivity_simple = metrics_data["prop_feed_offspring"].mean()

                results.append(
                    {
                        var_name: metrics_data[var_name].values[0],
                        "seed": metrics_data["seed"].values[0],
                        "n_agents": n_agents,
                        "feeding_benefit": feeding_benefit,
                        "amount_feeding": amount_feeding,
                        "feeding_selectivity": selectivity,
                        "feeding_selectivity_simple": selectivity_simple,
                        "avg_age_diff": avg_age_diff,
                    }
                )
            except Exception as e:
                continue

        results = pd.DataFrame(results)
        results.to_csv(os.path.join(expt_dir, "overall_results.csv"))

    results = results[results["feeding_benefit"].isna() == False]

    # put feeding selectivity on a 0-1 scale
    tmp = results["feeding_selectivity"].copy()
    results["feeding_selectivity"] = (tmp - tmp.min()) / (tmp.max() - tmp.min())
    for col in ["amount_feeding", "feeding_selectivity"]:
        tmp = results[col].copy()
        results[f"log_{col}"] = np.log(tmp + 1)

    dfs.append(results)
    plot_regressions(results, var_name)

# do combined regression plots
combined_df = pd.concat(dfs, ignore_index=True)
plot_regressions(combined_df, "combined")
