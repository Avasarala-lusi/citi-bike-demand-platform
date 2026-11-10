#!/bin/bash
set -euo pipefail

# Define the start and end years
START_YEAR=2025
END_YEAR=2026
WAIT_SECONDS="${WAIT_SECONDS:-75}"

# Loop through years
for year in $(seq $START_YEAR $END_YEAR); do
    # Determine the range of months for the current year
    if [ "$year" -eq 2025 ]; then
        start_month=1
        end_month=12
    else
        # Only go up to May (5) for 2026
        start_month=1
        end_month=5
    fi

    # Loop through months
    for month in $(seq $start_month $end_month); do
        echo ">>> Triggering drop_month.sh for $year-$month"
        ./drop_month.sh "$year" "$month"
        # Silver and both Gold branches use local Spark. Leave enough time for
        # the asset-triggered jobs to finish so laptop memory is not overrun.
        sleep "$WAIT_SECONDS"
    done
done

echo "✅ All months processed."
