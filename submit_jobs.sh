#!/bin/bash

# List of number of buildings
BUILDING_COUNTS=(5)

# List of seeds
SEEDS=(123 456 789)

# List of algorithms
ALGORITHMS=("DDPG" "SAC" "TD3" "PPO")

TRAINING_TYPE="independent"

# Path to the training configuration file
CONFIG_FILE="src/training_config.json"

# Backup original config file
cp $CONFIG_FILE "${CONFIG_FILE}.bak"

# Loop over each combination of algorithm, building count, and seed
for BUILDINGS in "${BUILDING_COUNTS[@]}"
do
    for ALGORITHM in "${ALGORITHMS[@]}"
    do
        for SEED in "${SEEDS[@]}"
        do
            # Use jq to update the config file with the new values
            jq --argjson buildings "$BUILDINGS" --argjson seed "$SEED" --arg algorithm "$ALGORITHM" --arg training_type "$TRAINING_TYPE"\
                '.no_buildings = $buildings | .seed = $seed | .algorithm = $algorithm | .training_type = $training_type' \
                $CONFIG_FILE > tmp.$$.json && mv tmp.$$.json $CONFIG_FILE

            # Print the modified config file to check changes
            echo "Config for $TRAINING_TYPE $ALGORITHM, $BUILDINGS buildings, and seed $SEED:"
            cat $CONFIG_FILE

            # Submit the SLURM job
            echo "Submitting job for $TRAINING_TYPE $ALGORITHM agent, $BUILDINGS buildings, and seed $SEED"
            sh submit.sh $BUILDINGS $SEED $ALGORITHM $TRAINING_TYPE
            sleep 2m
        done
    done
done

# Restore the original config file
mv "${CONFIG_FILE}.bak" $CONFIG_FILE
