#!/bin/bash

# August dates are shifted three months back to May 2026.
START_DATE="2026-07-22"
END_DATE="2026-08-03"
DAG_ID="4_ml_batch_demand_score"

echo "🚀 Triggering $DAG_ID from $START_DATE to $END_DATE..."

start_ts=$(date -j -f "%Y-%m-%d" "$START_DATE" +%s)
end_ts=$(date -j -f "%Y-%m-%d" "$END_DATE" +%s)

curr_ts=$start_ts

while [ "$curr_ts" -le "$end_ts" ]; do
    day_str=$(date -j -f "%s" "$curr_ts" +%Y-%m-%d)

    for hour in {0..23}; do
        formatted_hour=$(printf "%02d" "$hour")
        target_ts="${day_str}T${formatted_hour}:00:00"

        echo "📅 Triggering: $target_ts"

        docker compose exec airflow-webserver airflow dags trigger \
            "$DAG_ID" \
            --conf "{\"real_now\": \"$target_ts\", \"months_back\": 3}"
    done

    curr_ts=$((curr_ts + 86400))
done

echo "✅ All triggers submitted for $START_DATE through $END_DATE."