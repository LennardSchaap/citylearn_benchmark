#!/bin/sh

# size equipment

python src/preprocess.py tx_travis_county_neighborhood summer size_equipment || exit 1

# set schema and work order

python src/preprocess.py tx_travis_county_neighborhood summer set_sb3_work_order travis || exit 1

