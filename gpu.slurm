#!/bin/bash
#SBATCH --mail-user="s1914839@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=15868M
#SBATCH --time=7-00:00:00
#SBATCH --partition=gpu-long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

# Set job name dynamically
#SBATCH --job-name=CityLearn_gpu_central_${BUILDINGS}_${SEED}
#SBATCH --output=CityLearn_gpu_central_${BUILDINGS}_${SEED}_%j.out

# Load required modules
module load ALICE/default
module load CUDA/12.3.2
module load Miniconda3/23.9.0-0
. /easybuild/software/Miniconda3/23.9.0-0/etc/profile.d/conda.sh
conda activate benv
export PATH=$HOME/data1/conda/envs/benv/bin:$PATH

echo "Running job for $1 buildings with seed $2"

nvidia-smi
echo "## Number of available CUDA devices: $CUDA_VISIBLE_DEVICES"

# Print current details
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
export CWD=$(pwd)
echo "[$SHELL] ## Current working directory: $CWD"

# Run the simulation (assumes config file is already set up correctly)
sh workflow/preprocess.sh
sh workflow/simulate_central.sh

echo "[$SHELL] ## Finished simulation"
