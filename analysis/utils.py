from collections import defaultdict
import itertools
import os
import sys
from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from statsmodels.tools.validation.validation import Any
from tqdm import tqdm

# set seaborn theme and matplotlib font sizes
sns.set_theme(context="paper", style="darkgrid")
# plt.rcParams.update(
#     {
#         "font.size": 16,
#         "axes.labelsize": 36,
#         "legend.fontsize": 14,
#         "xtick.labelsize": 32,
#         "ytick.labelsize": 32,
#         "axes.titlesize": 40,
#     }
# )
plt.rcParams.update(
    {
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "axes.titlesize": 30,
    }
)

def load_metrics_data(
    data_dir, run_name, last_n_steps=5e4, window=None
) -> Tuple[pd.DataFrame, str]:
    var_str, seed_str = run_name.split("-")[-2:]
    tmp = var_str.split("_")
    var_name, var_val = "_".join(tmp[:-1]).replace("env_", ""), float(tmp[-1])
    seed = int(seed_str.split("_")[-1])

    metrics_data = pd.read_csv(os.path.join(data_dir, run_name, "metrics_data.csv"))
    last_timestamp = metrics_data["timestep"].max()
    metrics_data = metrics_data[
        metrics_data["timestep"] >= last_timestamp - last_n_steps
    ]

    metrics_data[var_name] = var_val
    metrics_data["seed"] = seed
    metrics_data["excess_offspring_feeding"] = (
        metrics_data["prop_feed_offspring"] - metrics_data["prop_face_offspring"]
    )

    if window is not None:
        metrics_data = metrics_data.rolling(window=window).mean()
    return metrics_data, var_name

def load_event_data(data_dir, run_name, event_type):
    datasets = []
    for filename in os.listdir(os.path.join(data_dir, run_name)):
        if event_type in filename and filename.endswith(".csv"):
            datasets.append(
                {
                    "timestep": int(filename.split("_")[-1].replace(".csv", "")),
                    "data": pd.read_csv(os.path.join(data_dir, run_name, filename)),
                }
            )
    datasets.sort(key=lambda x: x["timestep"])
    return pd.concat([x["data"] for x in datasets], ignore_index=True)

def load_eval_data(data_dir, run_name):
    # helper func
    def perc_change(b, a):
        return (b - a) / a * 100

    datasets = []
    for filename in os.listdir(os.path.join(data_dir, run_name)):
        if "eval_results" in filename and filename.endswith(".npy"):
            ts = int(filename.split("_")[-1].replace(".npy", ""))
            arr = np.load(os.path.join(data_dir, run_name, filename))
            arr = arr.transpose((1, 0))

            results = {
                "agent_selectivity": perc_change(arr[:, 1], arr[:, 0]),
                "infant_selectivity": perc_change(arr[:, 2], arr[:, 1]),
                "offspring_selectivity": perc_change(arr[:, 3], arr[:, 1]),
                "infant_offspring_selectivity": perc_change(arr[:, 4], arr[:, 1]),
            }

            for k in results.keys():
                # # get top 10% of agents
                # _sorted = np.sort(results[k])
                # results[k] = np.mean(_sorted[-1000:])
                results[k] = np.mean(results[k])

            datasets.append({"timestep": ts, **results})

    # return {"offspring_selectivity": np.mean(datasets)}

    combined = {}
    for k in datasets[0].keys():
        combined[k] = np.mean([x[k] for x in datasets])
    return combined

def compute_feeding_stats(feeding_data, death_data, age_cutoff=None):
    # compute average age difference between feeder and feedee
    feeding_data["age_diff"] = feeding_data["feeder_age"] - feeding_data["feedee_age"]
    avg_age_diff = feeding_data["age_diff"].mean()

    # assign unique agent ids
    for k in ["feeder", "feedee"]:
        feeding_data[k] = feeding_data[k].astype(int)
        feeding_data[f"{k}_born"] = (
            feeding_data["timestep"] - feeding_data[f"{k}_age"]
        ).astype(int)
        feeding_data[f"{k}_id"] = (
            feeding_data[k].astype(str) + "_" + feeding_data[f"{k}_born"].astype(str)
        )

    if age_cutoff is not None:
        feeding_data = feeding_data[feeding_data["feedee_age"] < age_cutoff]

    offspring_feeding = feeding_data[feeding_data["to_offspring"] == 1]

    feeder_counts_all = feeding_data["feeder_id"].value_counts()
    feeder_counts_offspring = offspring_feeding["feeder_id"].value_counts()
    feeder_counts = pd.DataFrame(
        {
            "total": feeder_counts_all.astype(int),
            "offspring": feeder_counts_offspring.astype(int),
        }
    ).fillna(0)

    feedee_counts_all = feeding_data["feedee_id"].value_counts()
    feedee_counts_offspring = offspring_feeding["feedee_id"].value_counts()
    feedee_counts = pd.DataFrame(
        {
            "total": feedee_counts_all.astype(int),
            "offspring": feedee_counts_offspring.astype(int),
        }
    ).fillna(0)

    return feeding_data, feeder_counts, feedee_counts, avg_age_diff

def compute_times_fed_vs_life_exp(feedee_counts, death_data, age_cutoff=None):
    # get age of each agent at death
    le_data = defaultdict(list)
    for i in range(len(death_data)):
        timestep, agent_slot, age = (
            death_data.loc[i]["timestep"],
            death_data.loc[i]["agent"],
            death_data.loc[i]["age"],
        )
        aid = f"{int(agent_slot)}_{int(timestep - age)}"
        times_fed = feedee_counts.total.get(aid, 0)
        le_data[times_fed].append(int(age))

    return {k: np.mean(v) for k, v in le_data.items()}

def do_regression(data, x, y, add_const=True):
    X = data[x]
    Y = data[y]
    if add_const:
        X = sm.add_constant(X)

    model = sm.OLS(Y, X)
    res = model.fit()
    conf_int = res.conf_int()

    intercept = res.params["const"] if add_const else 0
    intercept_conf_int = conf_int.loc["const"].values if add_const else [0, 0]

    return {
        "slope": res.params[x],
        "slope_conf_int": conf_int.loc[x].values,
        "intercept": intercept,
        "intercept_conf_int": intercept_conf_int,
        "r2": res.rsquared,
    }

def do_line_plot(
    ax,
    data,
    x,
    y,
    plot_kwargs={},
):
    data = data.copy(deep=True)
    sns.lineplot(data=data, x=x, y=y, ax=ax, **plot_kwargs)

def do_regression_plot(
    ax,
    data,
    x,
    y,
    reg_results,
    scatter_kwargs={},
    line_kwargs={},
    annotate=True,
    x_boundaries=None,
):
    data = data.copy(deep=True)
    if "color" not in line_kwargs:
        line_kwargs["color"] = "green"

    sns.scatterplot(data=data, x=x, y=y, ax=ax, **scatter_kwargs)

    # plot regression line
    if x_boundaries is not None:
        x_min, x_max = x_boundaries
    else:
        x_min, x_max = data[x].min(), data[x].max()
    x_vals = np.linspace(x_min, x_max, 100)
    y_mid = reg_results["intercept"] + reg_results["slope"] * x_vals
    sns.lineplot(x=x_vals, y=y_mid, ax=ax, **line_kwargs)

    # plot 95% confidence intervals
    y_low, y_high = (
        reg_results["intercept_conf_int"][0]
        + reg_results["slope_conf_int"][0] * x_vals,
        reg_results["intercept_conf_int"][1]
        + reg_results["slope_conf_int"][1] * x_vals,
    )
    for y_ in [y_low, y_high]:
        ax.plot(x_vals, y_, color=line_kwargs["color"], alpha=0.1)
    ax.fill_between(x_vals, y_low, y_high, color=line_kwargs["color"], alpha=0.2)

    # annotate with r2 value
    if annotate:
        ax.annotate(
            f"$R^2 = {reg_results['r2']:.2f}$",
            xy=(0.1, 0.9),
            xycoords="axes fraction",
            size=26,
            weight="bold",
        )
