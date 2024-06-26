# EcoJAX
This repository aims to provide a simple and modular codebase to run experiments on large neural-agents ecological models using JAX.

<p align="center">
  <img src="assets/video.gif" alt="Title" width="60%"/>
</p>

# Installation

Clone the repository and create a virtual environment.
The repo work (at least) in python 3.10 to 3.12.

```bash
git clone git@github.com:tboulet/EcoJAX.git
cd EcoJAX
python -m venv venv
source venv/bin/activate   # on Windows, use `venv\Scripts\activate.bat`
```

### Install JAX

On linux (note your CUDA version may vary) :
```bash
pip install jax[cuda12_pip]==0.4.24 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

On Windows :
```bash
pip install jax[cpu]==0.4.24
```

### Install the requirements

```bash
pip install -r requirements.txt
```

### Eventually install the package in editable mode

```bash
pip install -e .
```


# Usage

To run any experiment, run the ``run.py`` file with the desired configuration. We use Hydra as our configuration manager.

```bash
python run.py
```

To use a specific configuration, use the ``--config-name`` argument.

```bash
python run.py --config-name=my_config
```
 
You can modify a specific argument thanks to Hydra's override system.

```bash
python run.py path.to.arg.in.config=new_value
```

In particular, the 3 main components interacting with each other in the experiments are the environment, the agents species (or agents) and the model. You can modify the configuration of each of these components. The meaning of those three components is explained below.

```bash
python run.py env=gridworld agents=ne model=cnn
```

# Components

The environment, the agents and the model interact with each other in the context of $n$ agents living in an environment. Note that $n$ is not the current number of agents but the maximal number of agents in the simulation, and stay fixed during the whole simulation. This is a design choice to allow for efficient batching of the various arrays which is necessary for JAX parallelism.

## Environment

The environment is the world in which the agents evolve. It receives as input a batch of $n$ actions and returns two things :

- a batch of $n$ observations, one for each agent
- the "eco-information" that contains which agents just died, which agents are just born (and from which parents), etc.

The only environment currently available is the ``gridworld`` environment (and variations).

## Agents Species

The agents species is a class that contains the logic of the agents. It receives as input a batch of $n$ observations and returns a batch of $n$ actions. It also should manage internally the way the agents inherit their genes.

Agents species currently available are :
- ``ne`` (Neural Evolution) : the agents are neural networks that are evolved using a genetic algorithm

## Model

The model is the neural network architecture that will be used by the agents species.

Models currently available are :
- ``mlp`` (Multi-Layer Perceptron) : flatten and concatenate the observations and pass them through a simple feed-forward neural network
- ``cnn`` (Convolutional Neural Network) : do the same as the MLP but image-like observations are passed through a convolutional neural network
- ``random`` : the model is a random function that returns random actions
