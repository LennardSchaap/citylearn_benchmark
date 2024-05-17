#!/bin/sh

NUM_BUILDINGS=$(jq -r '.no_buildings' src/training_config.json)
ALGORITHM=$(jq -r '.algorithm' src/training_config.json)

# size equipment
python src/preprocess_central.py "tx_travis_county_neighborhood_${NUM_BUILDINGS}" size_equipment || exit 1

# set schema and work order

python src/preprocess_central.py "tx_travis_county_neighborhood_${NUM_BUILDINGS}" set_schema "tx_travis_county_neighborhood_${NUM_BUILDINGS}_${ALGORITHM}" || exit 1

