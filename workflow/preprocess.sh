#!/bin/sh

NUM_BUILDINGS="$1"
ALGORITHM="$2"


python src/preprocess.py "tx_travis_county_neighborhood_${NUM_BUILDINGS}" summer size_equipment || exit 1

# set schema and work order

python src/preprocess.py "tx_travis_county_neighborhood_${NUM_BUILDINGS}" summer set_sb3_work_order "tx_travis_county_neighborhood_${NUM_BUILDINGS}_${ALGORITHM}" || exit 1
