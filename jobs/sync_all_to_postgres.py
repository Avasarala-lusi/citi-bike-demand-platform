#!/usr/bin/env python3
"""Refresh the small operational PostgreSQL cache from Iceberg predictions."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from pyiceberg.table import StaticTable

from jobs.ml_utils import load_config
from jobs.sync_to_postgres import POSTGRES_COLUMNS


def main() -> None:
    cfg = load_config("/opt/airflow/config/local.yml")
    table_path = Path(cfg["catalog"]["warehouse_path"]) / "predictions" / "hourly_demand_predictions"
    frame = StaticTable.from_metadata(str(table_path)).scan().to_arrow().to_pandas()
    if frame.empty:
        raise ValueError("No Iceberg predictions are available to synchronize")
    frame = frame[POSTGRES_COLUMNS].copy()
    for column in ("prediction_hour_ts", "scored_at_ts", "feature_window_end_ts"):
        frame[column] = pd.to_datetime(frame[column], utc=True)

    values = []
    for raw in frame.to_dict(orient="records"):
        row = []
        for column in POSTGRES_COLUMNS:
            value = raw[column]
            if isinstance(value, pd.Timestamp):
                value = value.to_pydatetime()
            elif hasattr(value, "item"):
                value = value.item()
            if pd.isna(value):
                value = None
            row.append(value)
        values.append(tuple(row))

    connection = psycopg2.connect(
        "host=operational_cache_postgres dbname=operational_cache "
        "user=cache_user password=cache_password port=5432"
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hourly_demand_predictions (
                    station_id VARCHAR, prediction_hour_ts TIMESTAMPTZ, scored_at_ts TIMESTAMPTZ,
                    model_name VARCHAR, model_version VARCHAR, predicted_class BIGINT,
                    predicted_label VARCHAR, prob_class_0 DOUBLE PRECISION,
                    prob_class_1 DOUBLE PRECISION, prob_class_2 DOUBLE PRECISION,
                    baseline_predicted_class BIGINT, mlflow_run_id VARCHAR,
                    feature_window_end_ts TIMESTAMPTZ, threshold_q25 DOUBLE PRECISION,
                    threshold_q75 DOUBLE PRECISION, penalty_version VARCHAR,
                    actual_class BIGINT, actual_ride_count BIGINT,
                    model_penalty DOUBLE PRECISION, baseline_penalty DOUBLE PRECISION,
                    PRIMARY KEY (station_id, prediction_hour_ts)
                )
            """)
            columns = ", ".join(POSTGRES_COLUMNS)
            updates = ", ".join(
                f"{column}=EXCLUDED.{column}"
                for column in POSTGRES_COLUMNS
                if column not in {"station_id", "prediction_hour_ts"}
            )
            execute_values(
                cursor,
                f"INSERT INTO hourly_demand_predictions ({columns}) VALUES %s "
                f"ON CONFLICT (station_id, prediction_hour_ts) DO UPDATE SET {updates}",
                values,
                page_size=500,
            )
        connection.commit()
    finally:
        connection.close()
    print(f"🟢 Synchronized {len(values)} prediction rows to PostgreSQL.")


if __name__ == "__main__":
    main()
