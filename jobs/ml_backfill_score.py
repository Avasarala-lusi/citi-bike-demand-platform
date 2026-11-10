#!/usr/bin/env python3
"""Efficiently score an hourly simulation range in one Spark session."""
from __future__ import annotations

import argparse
from datetime import datetime

import joblib
import pandas as pd

from jobs.ml_batch_score import (
    class_probabilities,
    cost_sensitive_predict_single,
    ensure_predictions_table_exists,
)
from jobs.ml_features import (
    LABEL_MAP,
    DemandThresholds,
    build_one_prediction_feature_row,
    calculate_penalty,
    load_station_history,
    shifted_simulation_time,
)
from jobs.ml_utils import create_spark_session, get_gold_ml_table, get_models_dir, load_config, read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/opt/airflow/config/local.yml")
    parser.add_argument("--start-real-now", required=True)
    parser.add_argument("--end-real-now", required=True)
    parser.add_argument("--months-back", type=int, default=2)
    args = parser.parse_args()

    cfg = load_config(args.config)
    spark = create_spark_session(cfg, app_name="CitiBikeRangeDemandScore")
    catalog = cfg.get("catalog", {}).get("spark_catalog_name", "nyc")
    predictions_table = f"{catalog}.predictions.hourly_demand_predictions"
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {catalog}.predictions")
    ensure_predictions_table_exists(spark, predictions_table)

    models_dir = get_models_dir(cfg)
    model = joblib.load(models_dir / "champion_model.pkl")
    meta = read_json(models_dir / "champion_metadata.json")
    station_id = str(meta["station_id"])
    thresholds = DemandThresholds(q25=float(meta["q25"]), q75=float(meta["q75"]))
    feature_columns = list(meta["features"])

    real_hours = pd.date_range(
        pd.Timestamp(args.start_real_now).floor("h"),
        pd.Timestamp(args.end_real_now).floor("h"),
        freq="h",
    )
    feature_hours = [
        shifted_simulation_time(real_now=ts, months_back=args.months_back)
        for ts in real_hours
    ]
    maximum_target = max(feature_hours) + pd.Timedelta(hours=1)
    history = load_station_history(
        spark.table(get_gold_ml_table(cfg)), station_id=station_id, end_ts=maximum_target
    )

    records = []
    for feature_hour in feature_hours:
        features, row_meta = build_one_prediction_feature_row(
            history_pdf=history,
            station_id=station_id,
            feature_window_end_ts=feature_hour,
            thresholds=thresholds,
            feature_columns=feature_columns,
        )
        predicted_class = cost_sensitive_predict_single(model, features)
        p0, p1, p2 = class_probabilities(model, features)
        actual_class = row_meta["actual_class"]
        actual_count = row_meta["actual_ride_count"]
        baseline_class = int(row_meta["baseline_predicted_class"])
        records.append({
            "station_id": station_id,
            "prediction_hour_ts": row_meta["prediction_hour_ts"].to_pydatetime(),
            "scored_at_ts": datetime.utcnow(),
            "model_name": str(meta["model_name"]),
            "model_version": str(meta.get("model_version", "citibike-local-v1")),
            "predicted_class": predicted_class,
            "predicted_label": LABEL_MAP[predicted_class],
            "prob_class_0": p0,
            "prob_class_1": p1,
            "prob_class_2": p2,
            "baseline_predicted_class": baseline_class,
            "mlflow_run_id": str(meta.get("mlflow_run_id")),
            "feature_window_end_ts": row_meta["feature_window_end_ts"].to_pydatetime(),
            "threshold_q25": thresholds.q25,
            "threshold_q75": thresholds.q75,
            "penalty_version": str(meta.get("penalty_version", "asymmetric-v1")),
            "actual_class": int(actual_class) if actual_class is not None else None,
            "actual_ride_count": int(actual_count) if actual_count is not None else None,
            "model_penalty": calculate_penalty(actual_class, predicted_class),
            "baseline_penalty": calculate_penalty(actual_class, baseline_class),
        })

    result = pd.DataFrame(records)
    start_target = result["prediction_hour_ts"].min()
    end_target = result["prediction_hour_ts"].max()
    escaped_station = station_id.replace("'", "''")
    spark.sql(f"""
        DELETE FROM {predictions_table}
        WHERE station_id='{escaped_station}'
          AND prediction_hour_ts >= CAST('{start_target:%Y-%m-%d %H:%M:%S}' AS TIMESTAMP)
          AND prediction_hour_ts <= CAST('{end_target:%Y-%m-%d %H:%M:%S}' AS TIMESTAMP)
    """)
    prediction_schema = spark.table(predictions_table).schema
    spark.createDataFrame(result, schema=prediction_schema).writeTo(predictions_table).append()

    model_total = float(result["model_penalty"].sum())
    baseline_total = float(result["baseline_penalty"].sum())
    reduction = (baseline_total - model_total) / baseline_total * 100.0 if baseline_total else 0.0
    print(
        f"🟢 Stored {len(result)} hourly predictions for station {station_id}. "
        f"Model penalty={model_total:.0f}; last-week baseline={baseline_total:.0f}; "
        f"reduction={reduction:.2f}%"
    )


if __name__ == "__main__":
    main()
