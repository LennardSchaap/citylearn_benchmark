#!/bin/sh
python src/simulate_mappo.py simulate "/home/wortel/Documents/citylearn_benchmark/benchmark/data/neighborhoods/tx_travis_county_neighborhood_5/schema.json" || exit 1
