import sys
import logging
import subprocess
from datetime import datetime
from airflow.sdk import dag, task, Asset
from airflow.sdk.exceptions import AirflowFailException
from airflow.sdk.exceptions import AirflowSkipException

INCOMING_CITIBIKE_DATA = Asset(
    name="landing_zone/incoming_citibike_data",
    uri="file:///opt/airflow/data/incoming/citibike_data_drop",
)
BRONZE_CITIBIKE_DATA = Asset(
    name="warehouse/bronze/citibike_raw",
    uri="iceberg://bronze.citibike_raw",
)

@dag(
    dag_id="1_bronze_ingestion",
    start_date=datetime(2026, 4, 1),
    schedule=[INCOMING_CITIBIKE_DATA],
    catchup=False,
    max_active_runs=1,
)
def bronze_ingestion_dag():

    @task(task_id="invoke_bronze_spark_job", outlets=[BRONZE_CITIBIKE_DATA])
    def invoke_bronze_spark_job(**context):
        log = logging.getLogger("airflow.task")

        triggering_events = context.get("triggering_asset_events", {}).get(INCOMING_CITIBIKE_DATA, [])
        if not triggering_events:
            raise AirflowFailException("Missing incoming Citi Bike asset event metadata.")

        manifest_meta = dict(triggering_events[0].extra)
        run_id = context["run_id"]

        cmd = [
            sys.executable, "/opt/airflow/jobs/bronze_ingest.py",
            "--config", "/opt/airflow/config/local.yml",
            "--source_filepath", str(manifest_meta["source_filepath"]),
            "--source_filename", str(manifest_meta["source_filename"]),
            "--source_year", str(manifest_meta["source_year"]),
            "--source_month", str(manifest_meta["source_month"]),
            "--run_id", str(run_id),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        log.info("Bronze Spark job stdout:\n%s", result.stdout)

        if result.returncode != 0:
            log.error("Bronze Spark job stderr:\n%s", result.stderr)
            raise AirflowFailException(f"Bronze Spark job failed with exit code {result.returncode}")

        if "Idempotency skip" in result.stdout:
            log.info("🛑 Spark job detected a duplicate drop. Short-circuiting Airflow context to prevent downstream triggers.")
            raise AirflowSkipException("Skipping downstream updates because this data file has already been processed.")

        manifest_meta["ingestion_run_id"] = run_id
        context["outlet_events"][BRONZE_CITIBIKE_DATA].extra = manifest_meta

    invoke_bronze_spark_job()

bronze_ingestion_dag()
