from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from feature_engine.creation import CyclicalFeatures
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F


REQUIRED_FEATURES = [
    "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_168h",
    "rolling_3h_mean", "rolling_6h_mean", "rolling_6h_std",
    "rolling_24h_mean", "rolling_24h_std",
    "demand_velocity_1h", "demand_acceleration_1h",
    "delta_vs_24h", "delta_vs_168h", "short_long_mean_gap",
    "hour_of_day_sin", "hour_of_day_cos", "day_of_week_sin", "day_of_week_cos",
    "is_weekend", "weekday_morning_rush", "weekday_evening_rush", "is_transition_hour",
]

LABEL_MAP = {0: "Low", 1: "Normal", 2: "High"}

# TODO: SCORE EACH PREDICTION

# COST_MATRIX[actual_class, predicted_class]. Missing high demand is most expensive.
COST_MATRIX = np.array(
    [[0.0, 10.0, 20.0], [15.0, 0.0, 10.0], [30.0, 15.0, 0.0]],
    dtype=float,
)


@dataclass(frozen=True)
class DemandThresholds:
    q25: float
    q75: float


def select_highest_volume_station(spark_df: SparkDataFrame) -> str:
    row = (
        spark_df.groupBy("station_id")
        .agg(F.sum("ride_count").alias("total_rides"))
        .orderBy(F.col("total_rides").desc())
        .first()
    )
    if row is None:
        raise ValueError("Gold ML observation table is empty")
    return str(row["station_id"])


def load_station_history(
    spark_df: SparkDataFrame,
    station_id: str,
    end_ts: str | datetime | None = None,
) -> pd.DataFrame:
    frame = spark_df.filter(F.col("station_id") == str(station_id))
    if end_ts is not None:
        frame = frame.filter(F.col("station_hour_ts") <= F.lit(pd.Timestamp(end_ts).to_pydatetime()))
    pdf = frame.select("target_year", "target_month", "station_id", "station_name", "station_hour_ts", "ride_count").toPandas()
    if pdf.empty:
        raise ValueError(f"No Gold observations for station_id={station_id}")
    pdf["station_hour_ts"] = pd.to_datetime(pdf["station_hour_ts"])
    pdf = pdf.sort_values("station_hour_ts").drop_duplicates("station_hour_ts", keep="last")

    # Gold stores observed hours sparsely; explicitly restore zero-demand hours.
    full_hours = pd.date_range(pdf["station_hour_ts"].min(), pdf["station_hour_ts"].max(), freq="h")
    latest_name = str(pdf["station_name"].dropna().iloc[-1]) if pdf["station_name"].notna().any() else str(station_id)
    dense = pdf.set_index("station_hour_ts").reindex(full_hours)
    dense.index.name = "station_hour_ts"
    dense["station_id"] = str(station_id)
    dense["station_name"] = dense["station_name"].ffill().bfill().fillna(latest_name)
    dense["ride_count"] = dense["ride_count"].fillna(0).astype(int)
    dense["target_year"] = dense.index.year
    dense["target_month"] = dense.index.month
    return dense.reset_index()


def add_calendar_features(pdf: pd.DataFrame) -> pd.DataFrame:
    pdf = pdf.copy()
    ts = pd.to_datetime(pdf["station_hour_ts"])
    pdf["hour_of_day"] = ts.dt.hour.astype(int)
    pdf["day_of_week"] = ts.dt.dayofweek.astype(int)
    pdf["is_weekend"] = pdf["day_of_week"].isin([5, 6]).astype(int)
    weekday = pdf["day_of_week"] < 5
    pdf["weekday_morning_rush"] = (weekday & pdf["hour_of_day"].between(6, 9)).astype(int)
    pdf["weekday_evening_rush"] = (weekday & pdf["hour_of_day"].between(15, 19)).astype(int)
    pdf["is_transition_hour"] = (weekday & pdf["hour_of_day"].isin([5, 6, 9, 10, 14, 15, 19, 20])).astype(int)
    return pdf

# TODO: ADD ROLLING FEATURES
def add_lag_rolling_features(pdf: pd.DataFrame) -> pd.DataFrame:
    pdf = pdf.sort_values("station_hour_ts").copy()
    s = pdf["ride_count"].astype(float)
    for hours in (1, 2, 3, 24, 168):
        pdf[f"lag_{hours}h"] = s.shift(hours)
    shifted = s.shift(1)
    pdf["rolling_3h_mean"] = shifted.rolling(3, min_periods=3).mean()
    pdf["rolling_6h_mean"] = shifted.rolling(6, min_periods=6).mean()
    pdf["rolling_6h_std"] = shifted.rolling(6, min_periods=6).std()
    pdf["rolling_24h_mean"] = shifted.rolling(24, min_periods=24).mean()
    pdf["rolling_24h_std"] = shifted.rolling(24, min_periods=24).std()
    pdf["demand_velocity_1h"] = pdf["lag_1h"] - pdf["lag_2h"]
    pdf["demand_acceleration_1h"] = pdf["lag_1h"] - 2 * pdf["lag_2h"] + pdf["lag_3h"]
    pdf["delta_vs_24h"] = pdf["lag_1h"] - pdf["lag_24h"]
    pdf["delta_vs_168h"] = pdf["lag_1h"] - pdf["lag_168h"]
    pdf["short_long_mean_gap"] = pdf["rolling_3h_mean"] - pdf["rolling_24h_mean"]
    return pdf


def add_cyclical_features(pdf: pd.DataFrame) -> pd.DataFrame:
    cyc = CyclicalFeatures(
        variables=["hour_of_day", "day_of_week"],
        max_values={"hour_of_day": 24, "day_of_week": 7},
        drop_original=True,
    )
    return cyc.fit_transform(pdf.copy())


def build_training_features(pdf: pd.DataFrame) -> pd.DataFrame:
    result = add_calendar_features(pdf)
    result = add_lag_rolling_features(result)
    result = add_cyclical_features(result)
    return result.dropna(subset=REQUIRED_FEATURES).reset_index(drop=True)

# TODO: KEEP DATA SPLIT AS TIME ORDER
def chronological_split(pdf: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    pdf = pdf.sort_values("station_hour_ts").reset_index(drop=True)
    if len(pdf) < 10:
        raise ValueError("Not enough rows for chronological split")
    train_end = int(len(pdf) * train_frac)
    val_end = int(len(pdf) * (train_frac + val_frac))
    return pdf.iloc[:train_end].copy(), pdf.iloc[train_end:val_end].copy(), pdf.iloc[val_end:].copy()


def fit_thresholds(train_df: pd.DataFrame) -> DemandThresholds:
    return DemandThresholds(
        q25=float(train_df["ride_count"].quantile(0.25)),
        q75=float(train_df["ride_count"].quantile(0.75)),
    )


def class_from_ride_count(ride_count: float, thresholds: DemandThresholds) -> int:
    if ride_count < thresholds.q25:
        return 0
    if ride_count <= thresholds.q75:
        return 1
    return 2


def apply_thresholds(pdf: pd.DataFrame, thresholds: DemandThresholds) -> pd.DataFrame:
    result = pdf.copy()
    result["demand_class"] = result["ride_count"].apply(lambda value: class_from_ride_count(value, thresholds))
    return result


def prepare_model_frames(pdf: pd.DataFrame):
    features = build_training_features(pdf)
    train, validation, test = chronological_split(features)
    thresholds = fit_thresholds(train)
    return (
        apply_thresholds(train, thresholds),
        apply_thresholds(validation, thresholds),
        apply_thresholds(test, thresholds),
        thresholds,
        REQUIRED_FEATURES.copy(),
    )


def calculate_penalty(actual_class: int | None, predicted_class: int | None) -> float | None:
    if actual_class is None or predicted_class is None or pd.isna(actual_class) or pd.isna(predicted_class):
        return None
    return float(COST_MATRIX[int(actual_class), int(predicted_class)])


def mean_operational_penalty(y_true, y_pred) -> float:
    actual = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)
    return float(COST_MATRIX[actual, predicted].mean())


def shifted_simulation_time(real_now: str | datetime | None = None, months_back: int = 3) -> pd.Timestamp:
    ts = pd.Timestamp.utcnow().tz_localize(None) if real_now is None else pd.Timestamp(real_now).tz_localize(None) if pd.Timestamp(real_now).tzinfo else pd.Timestamp(real_now)
    return pd.Timestamp(ts.to_pydatetime() - relativedelta(months=months_back)).floor("h")


def build_one_prediction_feature_row(
    history_pdf: pd.DataFrame,
    station_id: str,
    feature_window_end_ts: str | datetime,
    thresholds: DemandThresholds,
    feature_columns: Iterable[str] = REQUIRED_FEATURES,
):
    feature_window_end_ts = pd.Timestamp(feature_window_end_ts).floor("h")
    prediction_hour_ts = feature_window_end_ts + pd.Timedelta(hours=1)
    pdf = history_pdf.copy().sort_values("station_hour_ts")
    pdf["station_hour_ts"] = pd.to_datetime(pdf["station_hour_ts"])
    history = pdf[pdf["station_hour_ts"] <= feature_window_end_ts]
    ride_by_ts = history.set_index("station_hour_ts")["ride_count"]

    def lag(hours: int) -> float:
        timestamp = prediction_hour_ts - pd.Timedelta(hours=hours)
        if timestamp not in ride_by_ts.index:
            raise ValueError(f"Missing required lag timestamp {timestamp}")
        return float(ride_by_ts.loc[timestamp])

    rolling_3 = [lag(i) for i in range(1, 4)]
    rolling_6 = [lag(i) for i in range(1, 7)]
    rolling_24 = [lag(i) for i in range(1, 25)]
    lag_1, lag_2, lag_3, lag_24, lag_168 = lag(1), lag(2), lag(3), lag(24), lag(168)
    row = pd.DataFrame([{
        "station_id": str(station_id), "station_hour_ts": prediction_hour_ts, "ride_count": np.nan,
        "lag_1h": lag_1, "lag_2h": lag_2, "lag_3h": lag_3, "lag_24h": lag_24, "lag_168h": lag_168,
        "rolling_3h_mean": float(np.mean(rolling_3)),
        "rolling_6h_mean": float(np.mean(rolling_6)), "rolling_6h_std": float(np.std(rolling_6, ddof=1)),
        "rolling_24h_mean": float(np.mean(rolling_24)), "rolling_24h_std": float(np.std(rolling_24, ddof=1)),
        "demand_velocity_1h": lag_1 - lag_2,
        "demand_acceleration_1h": lag_1 - 2 * lag_2 + lag_3,
        "delta_vs_24h": lag_1 - lag_24, "delta_vs_168h": lag_1 - lag_168,
        "short_long_mean_gap": float(np.mean(rolling_3) - np.mean(rolling_24)),
    }])
    row = add_cyclical_features(add_calendar_features(row))

    all_actuals = pdf.set_index("station_hour_ts")["ride_count"]
    actual_count = float(all_actuals.loc[prediction_hour_ts]) if prediction_hour_ts in all_actuals.index else None
    actual_class = class_from_ride_count(actual_count, thresholds) if actual_count is not None else None
    # The professor's current baseline is the same station and hour one week earlier.

    # TODO: USE DEMAND FROM THE SAME HOUR LAST WEEK
    baseline_class = class_from_ride_count(lag_168, thresholds)
    return row[list(feature_columns)], {
        "feature_window_end_ts": feature_window_end_ts,
        "prediction_hour_ts": prediction_hour_ts,
        "baseline_predicted_class": baseline_class,
        "actual_ride_count": actual_count,
        "actual_class": actual_class,
    }
