import os
import sys

sys.path.append(os.getcwd())


from gridworld import Gridworld

from utils import VideoWriter
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpy as np
import pickle
import datetime
from evojax.util import save_model, load_model
import yaml
from evojax.policy.base import PolicyState
from evojax.task.base import TaskState
from flax.struct import dataclass



@dataclass
class AgentStates_noparam(object):
    posx: jnp.uint16
    posy: jnp.uint16
    energy: jnp.ndarray
    time_good_level: jnp.uint16
    time_alive: jnp.uint16
    time_under_level: jnp.uint16
    alive: jnp.int8


@dataclass
class State_noparam(TaskState):
    last_actions: jnp.int8
    rewards: jnp.int8
    agents: AgentStates_noparam
    steps: jnp.int32

def state_no_params(state):
    agents=state.agents
    new_state_no_param=State_noparam(last_actions=state.last_actions, rewards=state.rewards,
          agents=AgentStates_noparam(posx=agents.posx, posy=agents.posy, energy=agents.energy, time_good_level=agents.time_good_level,
                                     time_alive=agents.time_alive, time_under_level=agents.time_under_level, alive=agents.alive),
          steps=state.steps)
    return new_state_no_param

state_no_params_fn=jax.jit(state_no_params)



def train(project_dir):
    with open(project_dir + "/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    max_steps = config["num_gens"] * config["gen_length"] + 1

    # initialize environment
    env = Gridworld(SX=config["grid_length"],
                    SY=config["grid_width"],
                    nb_agents=config["nb_agents"],
                    regrowth_scale=config["regrowth_scale"],
                    niches_scale=config["niches_scale"],
                    max_age=config["max_age"],
                    time_reproduce=config["time_reproduce"],
                    time_death=config["time_death"],
                    energy_decay=config["energy_decay"],
                    spontaneous_regrow=config["spontaneous_regrow"],
                    # wall_kill=config["wall_kill"],
                    )
    key = jax.random.PRNGKey(config["seed"])
    next_key, key = random.split(key)
    state = env.reset(next_key)

    # initialize policy


    gens = list(range(config["num_gens"]))
    
    nb_gens=config["num_gens"]
    

    for gen in gens:
        if(state.agents.alive.sum()==0):
            print("All the population died")
            break

        if gen % config["eval_freq"] == 0:
            vid = VideoWriter(project_dir + "/train/media/gen_" + str(gen) + ".mp4", 20.0)

        for i in range(config["gen_length"]):

                state, reward = env.step(state)
           
         

                if (gen % config["eval_freq"] == 0):
                    rgb_im = state.state[:, :, :3]
                    
                    rgb_im=jnp.clip(rgb_im,0,1)
                  
                
                     #white green and black
                    rgb_im=jnp.clip(rgb_im+jnp.expand_dims(state.state[:,:,1],axis=-1),0,1)
                    rgb_im=rgb_im.at[:,:,1].set(0)
                    rgb_im= 1-rgb_im
                    rgb_im=rgb_im-jnp.expand_dims(state.state[:,:,0],axis=-1)
                    
                    rgb_im = np.repeat(rgb_im, 2, axis=0)
                    rgb_im = np.repeat(rgb_im, 2, axis=1)
                    print(rgb_im.shape)
                    raise
                    vid.add(rgb_im)

                    

        if gen % config["eval_freq"] == 0:


            vid.close()

    



if __name__ == "__main__":
    project_dir = sys.argv[1]
    if not os.path.exists(project_dir + "/train/data"):
        os.makedirs(project_dir + "/train/data")

    if not os.path.exists(project_dir + "/train/models"):
        os.makedirs(project_dir + "/train/models")

    if not os.path.exists(project_dir + "/train/media"):
        os.makedirs(project_dir + "/train/media")
    train(project_dir)
