#!/bin/bash
#SBATCH --job-name=testing_td3_10_CityLearn
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="s1914839@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=10000M
#SBATCH --time=1:00:00
#SBATCH --partition=cpu-short
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

# load modules (assuming you start from the default environment)
# we explicitly call the modules to improve reproducibility
# in case the default settings change
module load ALICE/default
module load CUDA/12.3.2
module load Miniconda3/23.9.0-0
. /easybuild/software/Miniconda3/23.9.0-0/etc/profile.d/conda.sh
conda activate benv
export PATH=$HOME/data1/conda/envs/benv/bin:$PATH
module load slurm

which python

# Activate the conda env


module list

# nvidia-smi
# echo "## Number of available CUDA devices: $CUDA_VISIBLE_DEVICES"

echo "[$SHELL] #### Starting Python test"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

# Run the file
echo "[$SHELL] ## Run script"
python src/log_to_wandb.py
echo "[$SHELL] ## Script finished"

echo "[$SHELL] #### Finished Python test. Have a nice day"
