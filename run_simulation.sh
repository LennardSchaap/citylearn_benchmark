#!/bin/bash

CONFIG_FILE="src/training_config.json"

# Define paths based on config file variables
DATA_DIR=$(jq -r '.data_directory' "$CONFIG_FILE")
BUILDINGS=$(jq -r '.no_buildings' "$CONFIG_FILE")
SCHEMA_FILE="${DATA_DIR}/neighborhoods/tx_travis_county_neighborhood_${BUILDINGS}/schema.json"

# Preprocessing and simulation

sh workflow/preprocess.sh
echo "Preprocessed data."

if [ "$TRAINING_TYPE" = "independent" ]; then
    echo "Using work order file at: $WORK_ORDER_FILE"

    # Run the simulation using the work order file
    python src/simulate.py run_work_order "$WORK_ORDER_FILE" "$CONFIG_FILE" || exit 1
else
    # Run the simulation using the schema file
    python src/simulate.py simulate "$SCHEMA_FILE" "$CONFIG_FILE" || exit 1
fi
