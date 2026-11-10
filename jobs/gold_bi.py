import argparse

import yaml
from pyspark.sql import SparkSession


def month_bounds(year: int, month: int) -> tuple[str, str]:
    start = f"{year}-{month:02d}-01 00:00:00"
    end = f"{year + 1}-01-01 00:00:00" if month == 12 else f"{year}-{month + 1:02d}-01 00:00:00"
    return start, end


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    args = parser.parse_args()
    start_date, end_date = month_bounds(args.year, args.month)

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    catalog = cfg["catalog"]["spark_catalog_name"]
    spark = (
        SparkSession.builder.appName("CitiBikeGoldBI")
        .config(f"spark.sql.catalog.{catalog}", "org.apache.iceberg.spark.SparkCatalog")
        .config(f"spark.sql.catalog.{catalog}.type", cfg["catalog"]["type"])
        .config(f"spark.sql.catalog.{catalog}.warehouse", cfg["catalog"]["warehouse_path"])
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .getOrCreate()
    )
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {catalog}.gold")
    gold_table = f"{catalog}.gold.hourly_station_operations"
    silver_table = f"{catalog}.silver.citibike_trips"
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {gold_table} (
            target_year INT, target_month INT, station_hour_ts TIMESTAMP,
            station_id STRING, station_name STRING, ride_count BIGINT,
            avg_duration_minutes DOUBLE, avg_distance_km DOUBLE,
            member_share DOUBLE, ebike_share DOUBLE
        ) USING iceberg PARTITIONED BY (target_year, target_month)
    """)
    spark.sql(f"DELETE FROM {gold_table} WHERE target_year={args.year} AND target_month={args.month}")

    # TODO: GROUP RIDES BY HOUR AND STATION
    spark.sql(f"""
        INSERT INTO {gold_table}
        SELECT
            {args.year}, {args.month}, date_trunc('hour', started_at),
            start_station_id, max(start_station_name), count(*),
            avg(ride_duration_minutes), avg(trip_distance_km),
            avg(CASE WHEN member_casual='member' THEN 1.0 ELSE 0.0 END),
            avg(CASE WHEN lower(rideable_type) LIKE '%electric%' THEN 1.0 ELSE 0.0 END)
        FROM {silver_table}
        WHERE started_at >= timestamp('{start_date}') AND started_at < timestamp('{end_date}')
        GROUP BY date_trunc('hour', started_at), start_station_id
    """)
    print("🟢 Gold BI station operations mart rebuilt.")


if __name__ == "__main__":
    main()
