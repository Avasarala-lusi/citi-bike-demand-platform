import argparse
import hashlib
import sys

import yaml
from pyspark.sql import Row, SparkSession
from pyspark.sql.functions import col, current_timestamp, lit


SOURCE_COLUMN_TYPES = {
    "ride_id": "string",
    "rideable_type": "string",
    "started_at": "timestamp",
    "ended_at": "timestamp",
    "start_station_name": "string",
    "start_station_id": "string",
    "end_station_name": "string",
    "end_station_id": "string",
    "start_lat": "double",
    "start_lng": "double",
    "end_lat": "double",
    "end_lng": "double",
    "member_casual": "string",
}


def calculate_file_sha256(file_path: str) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source_filepath", required=True)
    parser.add_argument("--source_filename", required=True)
    parser.add_argument("--source_year", type=int, required=True)
    parser.add_argument("--source_month", type=int, required=True)
    parser.add_argument("--run_id", required=True)
    args = parser.parse_args()

    if not (2000 <= args.source_year <= 2100 and 1 <= args.source_month <= 12):
        raise ValueError("Invalid source year/month")

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)

    catalog = cfg["catalog"]["spark_catalog_name"]

    # TODO: checksum and duplicate protection
    checksum = calculate_file_sha256(args.source_filepath)
    print(f"🔒 File checksum: {checksum}")

    spark = (
        SparkSession.builder.appName("CitiBikeBronzeIngestion")
        .config(f"spark.sql.catalog.{catalog}", "org.apache.iceberg.spark.SparkCatalog")
        .config(f"spark.sql.catalog.{catalog}.type", cfg["catalog"]["type"])
        .config(f"spark.sql.catalog.{catalog}.warehouse", cfg["catalog"]["warehouse_path"])
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .getOrCreate()
    )

    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {catalog}.audit")
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {catalog}.bronze")
    ledger_table = f"{catalog}.audit.ingestion_ledger"
    bronze_table = f"{catalog}.bronze.citibike_raw"

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {ledger_table} (
            source_filename STRING, file_checksum STRING, ingestion_run_id STRING,
            status STRING, finished_at TIMESTAMP
        ) USING iceberg
    """)
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {bronze_table} (
            ride_id STRING, rideable_type STRING, started_at TIMESTAMP, ended_at TIMESTAMP,
            start_station_name STRING, start_station_id STRING,
            end_station_name STRING, end_station_id STRING,
            start_lat DOUBLE, start_lng DOUBLE, end_lat DOUBLE, end_lng DOUBLE,
            member_casual STRING,
            source_filename STRING, source_checksum STRING, source_year INT, source_month INT,
            ingestion_run_id STRING, ingested_at_utc TIMESTAMP
        ) USING iceberg PARTITIONED BY (source_year, source_month)
    """)

    if spark.table(ledger_table).where(
        (col("file_checksum") == checksum) & (col("status") == "SUCCESS")
    ).limit(1).count():
        print("🟢 Idempotency skip: this Citi Bike file already completed Bronze ingestion.")
        sys.exit(0)

    raw = spark.read.parquet(args.source_filepath)
    for name, data_type in SOURCE_COLUMN_TYPES.items():
        if name not in raw.columns:
            raw = raw.withColumn(name, lit(None).cast(data_type))

    bronze = raw.select(
        *[col(name).cast(data_type).alias(name) for name, data_type in SOURCE_COLUMN_TYPES.items()],
        lit(args.source_filename).alias("source_filename"),
        lit(checksum).alias("source_checksum"),
        lit(args.source_year).cast("int").alias("source_year"),
        lit(args.source_month).cast("int").alias("source_month"),
        lit(args.run_id).alias("ingestion_run_id"),
        current_timestamp().alias("ingested_at_utc"),
    )
    bronze.writeTo(bronze_table).append()

    ledger = spark.createDataFrame([
        Row(
            source_filename=args.source_filename,
            file_checksum=checksum,
            ingestion_run_id=args.run_id,
            status="SUCCESS",
        )
    ]).withColumn("finished_at", current_timestamp())
    ledger.writeTo(ledger_table).append()
    print(f"🟢 Bronze ingestion complete: {bronze.count():,} Citi Bike rides.")


if __name__ == "__main__":
    main()
