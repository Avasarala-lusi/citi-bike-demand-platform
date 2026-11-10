import argparse
import hashlib
import sys

import yaml
from pyspark.sql import Row, SparkSession, functions as F
from pyspark.sql.window import Window


def month_bounds(year: int, month: int) -> tuple[str, str]:
    start = f"{year}-{month:02d}-01 00:00:00"
    end = f"{year + 1}-01-01 00:00:00" if month == 12 else f"{year}-{month + 1:02d}-01 00:00:00"
    return start, end


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source_filename", required=True)
    parser.add_argument("--ingestion_run_id", required=True)
    parser.add_argument("--target_year", type=int, required=True)
    parser.add_argument("--target_month", type=int, required=True)
    args = parser.parse_args()
    start_date, end_date = month_bounds(args.target_year, args.target_month)

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    catalog = cfg["catalog"]["spark_catalog_name"]
    run_hash = hashlib.sha256(args.ingestion_run_id.encode()).hexdigest()[:16]
    candidate_table = f"{catalog}.silver.candidate_{run_hash}"

    spark = (
        SparkSession.builder.appName("CitiBikeSilverWAP")
        .config(f"spark.sql.catalog.{catalog}", "org.apache.iceberg.spark.SparkCatalog")
        .config(f"spark.sql.catalog.{catalog}.type", cfg["catalog"]["type"])
        .config(f"spark.sql.catalog.{catalog}.warehouse", cfg["catalog"]["warehouse_path"])
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .getOrCreate()
    )

    for namespace in ("audit", "bronze", "silver", "quarantine"):
        spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {catalog}.{namespace}")

    bronze_table = f"{catalog}.bronze.citibike_raw"
    silver_table = f"{catalog}.silver.citibike_trips"
    quarantine_table = f"{catalog}.quarantine.citibike_trips"
    ledger_table = f"{catalog}.audit.silver_publish_ledger"
    audit_table = f"{catalog}.audit.wap_audit_results"

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {silver_table} (
            ride_id STRING, rideable_type STRING, started_at TIMESTAMP, ended_at TIMESTAMP,
            start_station_name STRING, start_station_id STRING,
            end_station_name STRING, end_station_id STRING,
            start_lat DOUBLE, start_lng DOUBLE, end_lat DOUBLE, end_lng DOUBLE,
            member_casual STRING, ride_duration_minutes DOUBLE, trip_distance_km DOUBLE,
            source_filename STRING, source_checksum STRING, source_year INT, source_month INT,
            ingestion_run_id STRING, published_at_utc TIMESTAMP
        ) USING iceberg PARTITIONED BY (days(started_at))
    """)
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {quarantine_table} (
            ride_id STRING, rideable_type STRING, started_at TIMESTAMP, ended_at TIMESTAMP,
            start_station_name STRING, start_station_id STRING,
            end_station_name STRING, end_station_id STRING,
            start_lat DOUBLE, start_lng DOUBLE, end_lat DOUBLE, end_lng DOUBLE,
            member_casual STRING, source_filename STRING, source_checksum STRING,
            source_year INT, source_month INT, ingestion_run_id STRING,
            quarantine_reason STRING, wap_run_id STRING, quarantined_at TIMESTAMP
        ) USING iceberg
    """)
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {ledger_table} (
            ingestion_run_id STRING, source_filename STRING, status STRING, verified_at TIMESTAMP
        ) USING iceberg
    """)
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {audit_table} (
            ingestion_run_id STRING, source_filename STRING, source_year INT, source_month INT,
            total_rows BIGINT, valid_rows BIGINT, quarantined_rows BIGINT, duplicate_rows BIGINT,
            drop_rate DOUBLE, audit_status STRING, audited_at TIMESTAMP
        ) USING iceberg
    """)

    if spark.table(ledger_table).where(
        (F.col("ingestion_run_id") == args.ingestion_run_id) & (F.col("status") == "SUCCESS")
    ).limit(1).count():
        print("Idempotency Skip: Silver ledger already records SUCCESS.")
        sys.exit(0)

    raw = spark.table(bronze_table).where(F.col("ingestion_run_id") == args.ingestion_run_id)
    if not raw.limit(1).count():
        raise ValueError(f"No Bronze rows for ingestion_run_id={args.ingestion_run_id}")

    duplicate_count_window = Window.partitionBy("ride_id")
    checked = raw.withColumn("ride_id_occurrences", F.count("*").over(duplicate_count_window))

    # TODO: CHECK RIDE ID, VALID TIMES, START LOCATION, VALID RIDER TYPE
    valid_condition = (
        F.col("ride_id").isNotNull()
        & (F.trim(F.col("ride_id")) != "")
        & F.col("started_at").isNotNull()
        & F.col("ended_at").isNotNull()
        & F.col("start_station_id").isNotNull()
        & (F.trim(F.col("start_station_id")) != "")
        & F.col("start_station_name").isNotNull()
        & F.col("rideable_type").isNotNull()
        & F.col("member_casual").isin("member", "casual")
        & (F.col("ended_at").cast("long") > F.col("started_at").cast("long"))
        & ((F.col("ended_at").cast("long") - F.col("started_at").cast("long")) >= 60)
        & ((F.col("ended_at").cast("long") - F.col("started_at").cast("long")) <= 86400)
        & (F.col("started_at") >= F.lit(start_date).cast("timestamp"))
        & (F.col("started_at") < F.lit(end_date).cast("timestamp"))
        & F.col("start_lat").between(-90.0, 90.0)
        & F.col("start_lng").between(-180.0, 180.0)
        & (F.col("ride_id_occurrences") == 1)
    )

    duration = (
        (F.col("ended_at").cast("long") - F.col("started_at").cast("long")) / 60.0
    ).cast("double")
    has_end_coords = F.col("end_lat").between(-90.0, 90.0) & F.col("end_lng").between(-180.0, 180.0)
    distance = F.when(
        has_end_coords,
        F.expr(
            "6371.0 * 2.0 * asin(sqrt(pow(sin(radians(end_lat - start_lat) / 2.0), 2) + "
            "cos(radians(start_lat)) * cos(radians(end_lat)) * "
            "pow(sin(radians(end_lng - start_lng) / 2.0), 2)))"
        ),
    ).otherwise(F.lit(None).cast("double"))

    clean = checked.filter(valid_condition).select(
        "ride_id", "rideable_type", "started_at", "ended_at",
        "start_station_name", "start_station_id", "end_station_name", "end_station_id",
        "start_lat", "start_lng", "end_lat", "end_lng", "member_casual",
        duration.alias("ride_duration_minutes"), distance.alias("trip_distance_km"),
        "source_filename", "source_checksum", "source_year", "source_month",
        F.lit(args.ingestion_run_id).alias("ingestion_run_id"),
        F.current_timestamp().alias("published_at_utc"),
    )

    reason = (
        F.when(F.col("ride_id").isNull() | (F.trim(F.col("ride_id")) == ""), "missing_ride_id")
        .when(F.col("ride_id_occurrences") > 1, "duplicate_ride_id")
        .when(F.col("started_at").isNull() | F.col("ended_at").isNull(), "missing_timestamp")
        .when(F.col("start_station_id").isNull() | (F.trim(F.col("start_station_id")) == ""), "missing_start_station")
        .when(F.col("ended_at").cast("long") <= F.col("started_at").cast("long"), "invalid_time_order")
        .when(~F.col("member_casual").isin("member", "casual"), "invalid_member_type")
        .otherwise("failed_quality_contract")
    )

    # TODO: quarantine failed rows for reviewing
    quarantined = checked.filter(~valid_condition).select(
        "ride_id", "rideable_type", "started_at", "ended_at",
        "start_station_name", "start_station_id", "end_station_name", "end_station_id",
        "start_lat", "start_lng", "end_lat", "end_lng", "member_casual",
        "source_filename", "source_checksum", "source_year", "source_month", "ingestion_run_id",
        reason.alias("quarantine_reason"), F.lit(run_hash).alias("wap_run_id"),
        F.current_timestamp().alias("quarantined_at"),
    )

    total_count = checked.count()
    valid_count = clean.count()
    quarantine_count = total_count - valid_count
    duplicate_count = checked.filter(F.col("ride_id_occurrences") > 1).count()
    drop_rate = quarantine_count / total_count * 100.0 if total_count else 100.0
    # TODO: IF DROP RATE <=35% THEN FAIL TO AUDIT AND THE PROCESS STOPS.
    audit_status = "PASS" if total_count > 0 and drop_rate <= 35.0 else "FAIL"
    print(
        f"📊 WAP audit: total={total_count:,}, valid={valid_count:,}, "
        f"quarantined={quarantine_count:,}, duplicates={duplicate_count:,}, drop_rate={drop_rate:.2f}%"
    )

    audit_row = spark.createDataFrame([Row(
        ingestion_run_id=args.ingestion_run_id,
        source_filename=args.source_filename,
        source_year=args.target_year,
        source_month=args.target_month,
        total_rows=total_count,
        valid_rows=valid_count,
        quarantined_rows=quarantine_count,
        duplicate_rows=duplicate_count,
        drop_rate=drop_rate,
        audit_status=audit_status,
    )]).withColumn("audited_at", F.current_timestamp())
    audit_row.writeTo(audit_table).append()
    if audit_status != "PASS":
        raise ValueError(f"WAP gate rejected batch with {drop_rate:.2f}% invalid rows")

    spark.sql(f"DROP TABLE IF EXISTS {candidate_table}")
    try:
        clean.writeTo(candidate_table).using("iceberg").create()
        quarantined.writeTo(quarantine_table).append()
        spark.read.table(candidate_table).writeTo(silver_table).append()
        ledger = spark.createDataFrame([Row(
            ingestion_run_id=args.ingestion_run_id,
            source_filename=args.source_filename,
            status="SUCCESS",
        )]).withColumn("verified_at", F.current_timestamp())
        ledger.writeTo(ledger_table).append()
        print("🚀 WAP audit passed; trusted Citi Bike rows published to Silver.")
    finally:
        spark.sql(f"DROP TABLE IF EXISTS {candidate_table}")


if __name__ == "__main__":
    main()
