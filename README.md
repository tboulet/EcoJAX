# EcoJAX
Research project for studying ecosystems of neural agents. In the context of my internship at FLOWERS.

# Installation

Clone the repository and create a virtual environment.
The python version used was 3.12.2.

```bash
git clone git@github.com:tboulet/EcoJAX.git
cd EcoJAX
python -m venv venv
source venv/bin/activate
```

Install the dependencies. Note your CUDA version may vary.
```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install jax[cuda12_pip]==0.4.24 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
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

