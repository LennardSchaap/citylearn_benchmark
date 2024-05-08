#!/bin/sh

NUM_BUILDINGS="$1"
ALGORITHM="$2"

# size equipment
python src/preprocess_central.py "tx_travis_county_neighborhood_${NUM_BUILDINGS}" size_equipment || exit 1

# set schema and work order

python src/preprocess_central.py "tx_travis_county_neighborhood_${NUM_BUILDINGS}" set_schema "tx_travis_county_neighborhood_${NUM_BUILDINGS}_${ALGORITHM}" || exit 1

