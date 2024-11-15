#!/bin/bash
#SBATCH --job-name=CityLearn_cpu_independent_td3_10
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="s1914839@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=16064M
#SBATCH --time=03:00:00
#SBATCH --partition=cpu-medium
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1

# load modules (assuming you start from the default environment)
# we explicitly call the modules to improve reproducibility
# in case the default settings change
module load ALICE/default
module load CUDA/12.3.2
module load Miniconda3/23.9.0-0
. /easybuild/software/Miniconda3/23.9.0-0/etc/profile.d/conda.sh
conda activate benv
export PATH=$HOME/data1/conda/envs/benv/bin:$PATH
# module load slurm

echo "[$SHELL] #### Starting Python test"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

# Run the file
echo "[$SHELL] ## Run script"
sh workflow/preprocess.sh
sh workflow/simulate.sh
echo "[$SHELL] ## Script finished"

echo "[$SHELL] #### Finished Python test. Have a nice day"
