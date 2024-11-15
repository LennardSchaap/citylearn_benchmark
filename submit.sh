#!/bin/bash

# Get the arguments
BUILDINGS=$1
SEED=$2
ALGORITHM=$3
TRAINING_TYPE=$4
CONFIG_FILE=$5

# Define paths based on config file variables
DATA_DIR=$(jq -r '.data_directory' "$CONFIG_FILE")
SCHEMA_FILE="${DATA_DIR}/neighborhoods/tx_travis_county_neighborhood_${BUILDINGS}/schema.json"

# Create a temporary SLURM job script
JOB_SCRIPT="slurm_job_${TRAINING_TYPE}_${BUILDINGS}_${SEED}_${ALGORITHM}.sh"

# Location of the work order file for independent training
WORK_ORDER_FILE="workflow/work_order/tx_travis_county_neighborhood_${BUILDINGS}_${ALGORITHM}.sh"

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

# Set job name
#SBATCH --job-name=CityLearn_gpu_${TRAINING_TYPE}_${ALGORITHM}_${BUILDINGS}_${SEED}
#SBATCH --output=CityLearn_gpu_${TRAINING_TYPE}_${ALGORITHM}_${BUILDINGS}_${SEED}_%j.out

# Load required modules
module load ALICE/default
module load CUDA/12.3.2
module load Miniconda3/23.9.0-0
. /easybuild/software/Miniconda3/23.9.0-0/etc/profile.d/conda.sh
conda activate benv
export PATH=\$HOME/data1/conda/envs/benv/bin:\$PATH

echo "Running job for $BUILDINGS buildings with seed $SEED using algorithm $ALGORITHM and training type $TRAINING_TYPE"

nvidia-smi
echo "## Number of available CUDA devices: \$CUDA_VISIBLE_DEVICES"

# Preprocessing and simulation
if [ "$TRAINING_TYPE" = "independent" ]; then
    echo "Using work order file at: $WORK_ORDER_FILE"

    # Run the simulation using the work order file
    python src/simulate.py run_work_order "$WORK_ORDER_FILE" "$CONFIG_FILE" || exit 1
else
    # Run the simulation using the schema file
    python src/simulate.py simulate "$SCHEMA_FILE" "$CONFIG_FILE" || exit 1
fi

echo "Job complete for $TRAINING_TYPE training with $BUILDINGS buildings and algorithm $ALGORITHM."

EOT

# Submit the SLURM job
sbatch $JOB_SCRIPT

# Optionally remove the job script after submission
rm $JOB_SCRIPT
