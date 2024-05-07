#!/bin/sh

# size equipment
EXPERIMENT_NAME="$1"

python src/preprocess_central.py $EXPERIMENT_NAME tx_travis_county_neighborhood_5 size_equipment || exit 1

# set schema and work order

python src/preprocess_central.py $EXPERIMENT_NAME tx_travis_county_neighborhood_5 set_schema travis-resstock-2 || exit 1

