from collections import defaultdict
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

n_agents_initial = 1000
n_agents_max = 1000
data_dir = "logs/[NeuroEvolutionAgents_CNN_Gridworld]_09th08mo_15h03min59s_seed291"

feeding_datasets, birth_datasets, death_datasets = [], [], []
for filename in os.listdir(data_dir):
    if not filename.endswith(".csv"):
        continue
    obj = {
        "timestep": int(filename.split("_")[-1].replace(".csv", "")),
        "data": pd.read_csv(os.path.join(data_dir, filename)),
    }
    if "feed" in filename:
        feeding_datasets.append(obj)
    elif "birth" in filename:
        birth_datasets.append(obj)
    elif "death" in filename:
        death_datasets.append(obj)

# sort datasets by timestep
feeding_datasets.sort(key=lambda x: x["timestep"])
birth_datasets.sort(key=lambda x: x["timestep"])
death_datasets.sort(key=lambda x: x["timestep"])

# concat datasets
feeding_data = pd.concat([x["data"] for x in feeding_datasets], ignore_index=True)
birth_data = pd.concat([x["data"] for x in birth_datasets], ignore_index=True)
death_data = pd.concat([x["data"] for x in death_datasets], ignore_index=True)

# assign unique agent ids
for k in ["feeder", "feedee"]:
    feeding_data[f"{k}_born"] = feeding_data["timestep"] - feeding_data[f"{k}_age"]
    feeding_data[f"{k}_id"] = [
        f"{feeding_data.loc[i][k]}_{feeding_data.loc[i][f'{k}_born']}"
        for i in range(len(feeding_data))
    ]

# get number of times each agent feeds their offspring and is fed by their parent
feeders = np.unique(feeding_data["feeder_id"].values)
num_times_agent_feeds = {
    f_id: len(feeding_data[feeding_data["feeder_id"] == f_id]) for f_id in feeders
}
feedees = np.unique(feeding_data["feedee_id"].values)
num_times_agent_is_fed = {
    f_id: len(feeding_data[feeding_data["feedee_id"] == f_id]) for f_id in feedees
}

# get age of each agent at death
death_ages = {}
for i in range(len(death_data)):
    timestep, agent_slot, age = (
        death_data.loc[i]["timestep"],
        death_data.loc[i]["agent"],
        death_data.loc[i]["age"],
    )
    agent_id = f"{agent_slot}_{timestep - age}"
    death_ages[agent_id] = int(age)

times_fed_death_ages = {"times_fed": [], "death_age": []}
for agent_id, death_age in death_ages.items():
    times_fed = num_times_agent_feeds.get(agent_id, 0)
    times_fed_death_ages["times_fed"].append(times_fed)
    times_fed_death_ages["death_age"].append(death_age)
times_fed_death_ages = pd.DataFrame(times_fed_death_ages)



# print("feed:")
# print(feeding_data)
# print("birth:")
# print(birth_data.head)
# print("death:")
# print(death_data.head)
