from pyspark.sql import SparkSession, functions as F

spark = (
    SparkSession.builder.appName("CitiBikeGoldCoverageAudit")
    .config("spark.sql.catalog.nyc", "org.apache.iceberg.spark.SparkCatalog")
    .config("spark.sql.catalog.nyc.type", "hadoop")
    .config("spark.sql.catalog.nyc.warehouse", "/opt/airflow/data/warehouse")
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
    .getOrCreate()
)
gold = spark.table("nyc.gold.hourly_station_demand")
report = (
    gold.withColumn("year_month", F.concat(F.col("target_year"), F.lit("-"), F.lpad("target_month", 2, "0")))
    .groupBy("year_month")
    .agg(F.count("*").alias("station_hour_rows"), F.sum("ride_count").alias("rides"))
    .orderBy("year_month")
)
print("📊 Citi Bike Gold coverage (expected 2025-01 through 2026-05):")
report.show(30, truncate=False)
actual = {row["year_month"] for row in report.collect()}
expected = {f"{year}-{month:02d}" for year in (2025, 2026) for month in range(1, (13 if year == 2025 else 6))}
missing = sorted(expected - actual)
if missing:
    raise ValueError(f"Missing Gold months: {missing}")
print("🟢 Gold coverage audit passed for all 17 months.")
