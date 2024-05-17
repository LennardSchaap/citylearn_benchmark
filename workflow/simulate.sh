# Extract the value of 'no_buildings' key from JSON file using jq
NUM_BUILDINGS=$(jq -r '.no_buildings' src/training_config.json)

# Extract the value of 'algorithm' key from JSON file using jq
ALGORITHM=$(jq -r '.algorithm' src/training_config.json)

# Define the file path
WORK_ORDER_FILE="workflow/work_order/tx_travis_county_neighborhood_${NUM_BUILDINGS}_${ALGORITHM}.sh"

#!/bin/sh
python src/simulate.py run_work_order "$WORK_ORDER_FILE" || exit 1
