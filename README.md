This code uses the source code of `A framework for the design of representative neighborhoods for energy flexibility assessment in CityLearn` paper as a basis to generate CityLearn environments of different scale. These environments are used as a basis for benchmarking RL algorithms in environments of increasing complexity, allowing the comparison of single-agent and multi-agent algorithms across different district sizes. See my thesis: "Benchmarking Reinforcement Learning Algorithms for Demand Response in Urban Energy Systems " https://theses.liacs.nl/pdf/2023-2024-SchaapLLennard.pdf"

## Installation
First, clone this repository as:

```bash
git clone --recurse-submodules https://github.com/LennardSchaap/citylearn-rl-benchmark
git submodule update --init --recursive
```
Make a new virtual environment with Python3.9
```bash
python3.9 -m venv env
source env/bin/activate
```
Install the dependencies in [requirements.txt](requirements.txt), using Python 3.8:
```bash
pip install -r requirements.txt
```

## Running simulations

Set the parameters of the run in src/training_config.json:

For example:

"training_type" : "central",
"log_to_wandb" : false,
"no_buildings" : 50,
"algorithm" : "DDPG",
"policy_type" : "MlpPolicy",
"seed" : 42,
"episodes" : 600,
"n_layers" : 3,
"neurons_per_layer" : 1024,
"frame-stack-ppo" : false,
"n_stack" : 4,
"device" : "cuda",
"use_dhw_storage" : true,
"use_electrical_storage" : true,
"model_save_freq" : 1,
"load_saved_model" : false,
"evaluate" : true,
"save_env_data_during_training" : false,
"data_directory" : "/.../citylearn-rl-benchmark/data",
"conda_environment" : "",
"virtual_environment_path" : "/.../citylearn-rl-benchmark/env"

Run the bash script
```bash
sh run_simulation.sh
```

## Results
The results of the simulation are stored in `data/simulation_output` that is automatically generated and has one subdirectory for each building that is simulated. This subdirectory has four output files:
1. `building_id-environment.csv`: Time series of static and runtime environment variables during training episodes and final evaluation.
2. `building_id-kpi.csv`: Energy flexibility KPIs calculated at the end of each episode during training episodes and final evaluation.
3. `building_id-reward.csv`: Time series of reward during training episodes and final evaluation.
4. `building_id-timer.csv`: Time it took for each episode to complete during training episodes and final evaluation.

The notebooks in the [analysis](analysis) directory can be used to produce the figures of the paper.



