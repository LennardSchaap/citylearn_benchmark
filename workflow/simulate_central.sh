#!/bin/sh

NUM_BUILDINGS="$1"
ALGORITHM="$2"

python src/simulate_central.py simulate "/home/wortel/Documents/citylearn_benchmark/benchmark/data/neighborhoods/tx_travis_county_neighborhood_${NUM_BUILDINGS}/schema.json" "$ALGORITHM" || exit 1
