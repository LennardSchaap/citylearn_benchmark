#!/bin/bash

# List of number of buildings
BUILDING_COUNTS=(5)

# List of seeds
SEEDS=(123)

# List of algorithms
ALGORITHMS=("DDPG")

TRAINING_TYPE="independent"

# Path to the training configuration file template
CONFIG_FILE_TEMPLATE="src/training_config.json"

# Preprocessing for independent agent training:

echo "Preprocessing data..."
for BUILDINGS in "${BUILDING_COUNTS[@]}"
do
    for ALGORITHM in "${ALGORITHMS[@]}"
    do
        # Size equipment
        python src/preprocess.py "tx_travis_county_neighborhood_${BUILDINGS}" summer size_equipment || exit 1

        # Set schema and work order
        python src/preprocess.py "tx_travis_county_neighborhood_${BUILDINGS}" summer set_sb3_work_order "tx_travis_county_neighborhood_${BUILDINGS}_${ALGORITHM}" || exit 1
    done
done
echo "done."
echo "Starting job submissions..."

# Loop over each combination of algorithm, building count, and seed
for BUILDINGS in "${BUILDING_COUNTS[@]}"
do
    for ALGORITHM in "${ALGORITHMS[@]}"
    do
        for SEED in "${SEEDS[@]}"
        do
            # Define a unique config file for each job
            CONFIG_FILE="src/training_config_${TRAINING_TYPE}_${ALGORITHM}_${BUILDINGS}_${SEED}.json"

            # Use jq to update the template config file and save it as a unique config file
            jq --argjson buildings "$BUILDINGS" --argjson seed "$SEED" --arg algorithm "$ALGORITHM" --arg training_type "$TRAINING_TYPE" \
                '.no_buildings = $buildings | .seed = $seed | .algorithm = $algorithm | .training_type = $training_type' \
                "$CONFIG_FILE_TEMPLATE" > "$CONFIG_FILE"

            # Print the modified config file to check changes
            echo "Config for $TRAINING_TYPE $ALGORITHM, $BUILDINGS buildings, and seed $SEED:"
            cat "$CONFIG_FILE"

            # Submit the SLURM job with the unique config file as an argument
            echo "Submitting job for $TRAINING_TYPE $ALGORITHM agent, $BUILDINGS buildings, and seed $SEED"
            sh submit.sh $BUILDINGS $SEED $ALGORITHM $TRAINING_TYPE "$CONFIG_FILE"
        done
    done
done
