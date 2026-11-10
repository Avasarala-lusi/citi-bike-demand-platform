#!/bin/bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: ./drop_month.sh YEAR MONTH"
    exit 1
fi

YEAR=$1
MONTH_NUM=$((10#$2))
MONTH=$(printf "%02d" "$MONTH_NUM")
TARGET_DIR="./data/incoming"
PARQUET_FILE="citibike_tripdata_${YEAR}-${MONTH}.parquet"

if [ ! -f "${TARGET_DIR}/${PARQUET_FILE}" ]; then
    echo "❌ Missing ${TARGET_DIR}/${PARQUET_FILE}"
    exit 1
fi

FILE_SIZE=$(stat -f%z "${TARGET_DIR}/${PARQUET_FILE}" 2>/dev/null || stat -c%s "${TARGET_DIR}/${PARQUET_FILE}")

cat > "${TARGET_DIR}/citibike_tripdata_${YEAR}_${MONTH}.manifest.json" <<EOF
{
  "dataset": "citibike_trip_history",
  "source_filepath": "/opt/airflow/data/incoming/${PARQUET_FILE}",
  "source_filename": "${PARQUET_FILE}",
  "source_year": ${YEAR},
  "source_month": ${MONTH_NUM},
  "file_size_bytes": ${FILE_SIZE},
  "discovered_at_utc": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF

echo "📡 Citi Bike manifest generated for ${PARQUET_FILE}."
