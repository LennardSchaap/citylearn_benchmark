#!/bin/bash

# Get the arguments
BUILDINGS=$1
SEED=$2
ALGORITHM=$3
TRAINING_TYPE=$4

# Create a temporary SLURM job script
JOB_SCRIPT="slurm_job_${TRAINING_TYPE}_${BUILDINGS}_${SEED}_${ALGORITHM}.sh"

cat <<EOT > $JOB_SCRIPT
#!/bin/bash
#SBATCH --mail-user="s1914839@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=15868M
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

# Set job name dynamically
#SBATCH --job-name=CityLearn_gpu_${TRAINING_TYPE}_${ALGORITHM}_${BUILDINGS}_${SEED}
#SBATCH --output=CityLearn_gpu_${TRAINING_TYPE}_${ALGORITHM}_${BUILDINGS}_${SEED}_%j.out

# Load required modules
module load ALICE/default
module load CUDA/12.3.2
module load Miniconda3/23.9.0-0
. /easybuild/software/Miniconda3/23.9.0-0/etc/profile.d/conda.sh
conda activate benv
export PATH=\$HOME/data1/conda/envs/benv/bin:\$PATH

echo "Running job for $BUILDINGS buildings with seed $SEED using algorithm $ALGORITHM"

nvidia-smi
echo "## Number of available CUDA devices: \$CUDA_VISIBLE_DEVICES"

# Print current details
echo "[$SHELL] ## This is \$SLURM_JOB_USER on \$HOSTNAME and this job has the ID \$SLURM_JOB_ID"
export CWD=\$(pwd)
echo "[$SHELL] ## Current working directory: \$CWD"

# Run the simulation (assumes config file is already set up correctly)
sh workflow/preprocess.sh
sleep 1m
sh workflow/simulate_${TRAINING_TYPE}.sh

echo "[$SHELL] ## Finished simulation"

EOT

# Submit the SLURM job
sbatch $JOB_SCRIPT

# Optionally remove the job script after submission
rm $JOB_SCRIPT
