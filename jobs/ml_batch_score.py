from __future__ import annotations

import argparse
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType, TimestampType

from jobs.ml_features import (
    COST_MATRIX, LABEL_MAP, DemandThresholds, build_one_prediction_feature_row,
    calculate_penalty, load_station_history, shifted_simulation_time,
)
from jobs.ml_utils import create_spark_session, get_gold_ml_table, get_models_dir, load_config, read_json


def cost_sensitive_predict_single(model: Any, features: pd.DataFrame) -> int:
    probabilities = model.predict_proba(features)
    complete = np.zeros((1, 3), dtype=float)
    for index, class_value in enumerate(model.classes_):
        complete[0, int(class_value)] = probabilities[0, index]
    return int((complete @ COST_MATRIX).argmin(axis=1)[0])


def class_probabilities(model: Any, features: pd.DataFrame):
    values = model.predict_proba(features)[0]
    mapping = {int(cls): float(prob) for cls, prob in zip(model.classes_, values)}
    return mapping.get(0), mapping.get(1), mapping.get(2)


def ensure_predictions_table_exists(spark: SparkSession, table_name: str) -> None:
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            station_id STRING, prediction_hour_ts TIMESTAMP, scored_at_ts TIMESTAMP,
            model_name STRING, model_version STRING, predicted_class INT, predicted_label STRING,
            prob_class_0 DOUBLE, prob_class_1 DOUBLE, prob_class_2 DOUBLE,
            baseline_predicted_class INT, mlflow_run_id STRING, feature_window_end_ts TIMESTAMP,
            threshold_q25 DOUBLE, threshold_q75 DOUBLE, penalty_version STRING,
            actual_class INT, actual_ride_count INT, model_penalty DOUBLE, baseline_penalty DOUBLE
        ) USING iceberg PARTITIONED BY (station_id)
        TBLPROPERTIES ('format-version'='2')
    """)


def run_batch_score(config_path: str | None, real_now: str | None, months_back: int) -> None:
    cfg = load_config(config_path)
    spark = create_spark_session(cfg, app_name="CitiBikeBatchDemandScore")
    catalog = cfg.get("catalog", {}).get("spark_catalog_name", "nyc")
    predictions_table = f"{catalog}.predictions.hourly_demand_predictions"
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {catalog}.predictions")
    ensure_predictions_table_exists(spark, predictions_table)

    models_dir = get_models_dir(cfg)
    model_path = models_dir / "champion_model.pkl"
    metadata_path = models_dir / "champion_metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Train the Citi Bike champion model before scoring")

    model = joblib.load(model_path)
    meta = read_json(metadata_path)
    station_id = str(meta["station_id"])
    thresholds = DemandThresholds(q25=float(meta["q25"]), q75=float(meta["q75"]))
    feature_columns = list(meta["features"])
    feature_window_end_ts = shifted_simulation_time(real_now=real_now, months_back=months_back)
    prediction_hour_ts = feature_window_end_ts + pd.Timedelta(hours=1)
    print(f"Station: {station_id}; feature window: {feature_window_end_ts}; target: {prediction_hour_ts}")

    history = load_station_history(
        spark.table(get_gold_ml_table(cfg)), station_id=station_id, end_ts=prediction_hour_ts
    )
    available_through = pd.to_datetime(history["station_hour_ts"]).max()
    if available_through < prediction_hour_ts:
        print(
            f"Data availability skip: target {prediction_hour_ts} is newer than "
            f"Gold actuals through {available_through}."
        )
        return
    live_features, row_meta = build_one_prediction_feature_row(
        history_pdf=history,
        station_id=station_id,
        feature_window_end_ts=feature_window_end_ts,
        thresholds=thresholds,
        feature_columns=feature_columns,
    )
    predicted_class = cost_sensitive_predict_single(model, live_features)
    p0, p1, p2 = class_probabilities(model, live_features)
    actual_class = row_meta["actual_class"]
    actual_count = row_meta["actual_ride_count"]
    baseline_class = int(row_meta["baseline_predicted_class"])
    model_penalty = calculate_penalty(actual_class, predicted_class)
    baseline_penalty = calculate_penalty(actual_class, baseline_class)

    schema = StructType([
        StructField("station_id", StringType()), StructField("prediction_hour_ts", TimestampType()),
        StructField("scored_at_ts", TimestampType()), StructField("model_name", StringType()),
        StructField("model_version", StringType()), StructField("predicted_class", IntegerType()),
        StructField("predicted_label", StringType()), StructField("prob_class_0", DoubleType()),
        StructField("prob_class_1", DoubleType()), StructField("prob_class_2", DoubleType()),
        StructField("baseline_predicted_class", IntegerType()), StructField("mlflow_run_id", StringType()),
        StructField("feature_window_end_ts", TimestampType()), StructField("threshold_q25", DoubleType()),
        StructField("threshold_q75", DoubleType()), StructField("penalty_version", StringType()),
        StructField("actual_class", IntegerType()), StructField("actual_ride_count", IntegerType()),
        StructField("model_penalty", DoubleType()), StructField("baseline_penalty", DoubleType()),
    ])
    record = {
        "station_id": station_id,
        "prediction_hour_ts": prediction_hour_ts.to_pydatetime(),
        "scored_at_ts": datetime.utcnow(),
        "model_name": str(meta["model_name"]),
        "model_version": str(meta.get("model_version", "citibike-local-v1")),
        "predicted_class": predicted_class,
        "predicted_label": LABEL_MAP[predicted_class],
        "prob_class_0": p0, "prob_class_1": p1, "prob_class_2": p2,
        "baseline_predicted_class": baseline_class,
        "mlflow_run_id": str(meta.get("mlflow_run_id")),
        "feature_window_end_ts": feature_window_end_ts.to_pydatetime(),
        "threshold_q25": thresholds.q25, "threshold_q75": thresholds.q75,
        "penalty_version": str(meta.get("penalty_version", "asymmetric-v1")),
        "actual_class": int(actual_class) if actual_class is not None else None,
        "actual_ride_count": int(actual_count) if actual_count is not None else None,
        "model_penalty": float(model_penalty) if model_penalty is not None else None,
        "baseline_penalty": float(baseline_penalty) if baseline_penalty is not None else None,
    }
    spark_record = spark.createDataFrame(pd.DataFrame([record]), schema=schema)
    escaped_station = station_id.replace("'", "''")
    spark.sql(f"""
        DELETE FROM {predictions_table}
        WHERE station_id='{escaped_station}'
          AND prediction_hour_ts=CAST('{prediction_hour_ts:%Y-%m-%d %H:%M:%S}' AS TIMESTAMP)
    """)
    spark_record.writeTo(predictions_table).append()
    print(
        f"🟢 Citi Bike prediction stored. ML={LABEL_MAP[predicted_class]}, "
        f"last-week baseline={LABEL_MAP[baseline_class]}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--real-now", default=None)
    parser.add_argument("--months-back", type=int, default=3)
    args = parser.parse_args()
    run_batch_score(args.config, args.real_now, args.months_back)


if __name__ == "__main__":
    main()
