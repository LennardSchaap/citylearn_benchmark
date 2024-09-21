#!/bin/sh

NUM_BUILDINGS=$(jq -r '.no_buildings' src/training_config.json)
ALGORITHM=$(jq -r '.algorithm' src/training_config.json)
WORK_ORDER_FILE="workflow/work_order/tx_travis_county_neighborhood_${NUM_BUILDINGS}_${ALGORITHM}.sh"

python src/simulate.py run_work_order "$WORK_ORDER_FILE" || exit 1
