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

_scatter_kwargs = {"s": 60, "alpha": 0.3}
_reg_line_kwargs = {"linewidth": 5}
colors = ["#D81C8D", "#5C1CCD"]
# colors = ["#D33668", "#E47500"]
scatter_plot_kwargs = {
    "enabled": {**_scatter_kwargs, "c": colors[0]},
    "disabled": {**_scatter_kwargs, "c": colors[1]},
}
regression_plot_kwargs = {
    "enabled": {**_reg_line_kwargs, "color": colors[0]},
    "disabled": {**_reg_line_kwargs, "color": colors[1]},
}

data_dir = "outputs"

def preprocess(df):
    tmp = df["feeding_selectivity"].copy()
    df["feeding_selectivity"] = (tmp - tmp.min()) / (tmp.max() - tmp.min())
    df = df[df["feeding_benefit"].isna() == False]
    for col in ["amount_feeding", "feeding_selectivity"]:
        tmp = df[col].copy()
        df[f"log_{col}"] = np.log(tmp + 1)
    return df

control_dfs = []
for var_name in os.listdir(os.path.join(data_dir, "control")):
    expt_dir = os.path.join(data_dir, "control", var_name)
    if not os.path.isdir(expt_dir):
        continue
    df = pd.read_csv(os.path.join(expt_dir, "overall_results.csv"))
    control_dfs.append(df)

non_control_dfs = []
for var_name in os.listdir(os.path.join(data_dir, "feeding")):
    expt_dir = os.path.join(data_dir, "feeding", var_name)
    if not os.path.isdir(expt_dir):
        continue
    df = pd.read_csv(os.path.join(expt_dir, "overall_results.csv"))
    non_control_dfs.append(df)

control_data = pd.concat(control_dfs, ignore_index=True)
non_control_data = pd.concat(non_control_dfs, ignore_index=True)

print("Control data:")
print(control_data.head())
print(len(control_data))

print("\nNon-control data:")
print(non_control_data.head())
print(len(non_control_data))

control_data["kin_recognition"] = "disabled"
non_control_data["kin_recognition"] = "enabled"
data = preprocess(pd.concat([control_data, non_control_data], ignore_index=True))

print("\nData:")
print(data.head())

fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
x_boundaries = data["feeding_benefit"].min(), data["feeding_benefit"].max()
for i, y_var in enumerate(["log_amount_feeding", "log_feeding_selectivity"]):
    for condition in ["enabled", "disabled"]:
        df = data[data["kin_recognition"] == condition]
        reg_results = do_regression(df, "feeding_benefit", y_var, add_const=True)
        do_regression_plot(
            axs[i],
            df,
            "feeding_benefit",
            y_var,
            reg_results,
            scatter_kwargs=scatter_plot_kwargs[condition],
            line_kwargs=regression_plot_kwargs[condition],
            annotate=False,
            x_boundaries=x_boundaries,
        )
        r2_pos = 0.9 if condition == "enabled" else 0.8
        axs[i].annotate(
            f"$(R^2 = {reg_results['r2']:.2f})$",
            xy=(0.1, r2_pos),
            xycoords="axes fraction",
            size=18,
            weight="bold",
        )
axs[0].set(title="", xlabel="", ylabel="log(1 + feeding amount)")
axs[1].set(title="", xlabel="Feeding benefit", ylabel="log(1 + feeding selectivity)")
fig.tight_layout()
fig.suptitle("Effect of kin recognition on the development of feeding behaviour")

for fmt in ["png", "svg"]:
    fig.savefig(os.path.join(data_dir, f"kin_recognition.{fmt}"))
