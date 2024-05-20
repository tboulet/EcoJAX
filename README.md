# EcoJAX
Research project for studying ecosystems of neural agents. In the context of my internship at FLOWERS.

# Installation

Clone the repository and create a virtual environment.
The python version used was 3.12.2.

```bash
git clone git@github.com:tboulet/EcoJAX.git
cd EcoJAX
python -m venv venv
source venv/bin/activate   # on Windows, use `venv\Scripts\activate.bat`
```

### Install JAX

On linux (note your CUDA version may var) :
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

Install EvoJAX (for now)
```bash
cd evojax
pip install -e .
cd ..
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

