#!/bin/bash

#SBATCH --job-name=jupyter_notebook
#SBATCH --mem=2G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu-short
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out

unset XDG_RUNTIME_DIR
module load IPython/7.7.0-foss-2019a-Python-3.7.2

echo "Running the notebook on $(hostname)"
IPADDR=$(hostname -i)

echo "#### Starting notebook"
srun jupyter notebook --ip=$IPADDR --no-browser --port=8989
echo "#### Terminated notebook. Done"
