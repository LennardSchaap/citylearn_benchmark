#!/bin/sh

NUM_BUILDINGS=$(jq -r '.no_buildings' src/training_config.json)
ALGORITHM=$(jq -r '.algorithm' src/training_config.json)
DATA_DIR=$(jq -r '.data_directory' src/training_config.json)

SCHEMA_FILE="${DATA_DIR}/neighborhoods/tx_travis_county_neighborhood_${NUM_BUILDINGS}/schema.json"

python src/simulate_central.py simulate "$SCHEMA_FILE" "$ALGORITHM" || exit 1
