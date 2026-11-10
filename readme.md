# Citi Bike Demand Prediction Platform

This local Docker Compose project adapts the course taxi architecture to official Citi Bike trip histories from January 2025 through May 2026. It combines an idempotent Medallion/WAP data pipeline, cost-sensitive demand modeling, PostgreSQL operational serving, FastAPI, and a Streamlit business dashboard.

## Verified local run (August 4, 2026)

- 17 official monthly ZIP archives and 17 normalized Parquet files
- 60,307,939 raw rides; 60,258,767 trusted Silver rides
- 49,172 rejected/quarantined rows (0.082%)
- Gold coverage passed for every month from 2025-01 through 2026-05
- Champion: Optuna-tuned LightGBM at highest-volume station `6140.05`
- 312-hour evaluation: model penalty 755 vs. last-week baseline 1,200
- Operational penalty reduction: **37.08%**

Penalty values are business-cost proxy units, not dollars. To make a dollar claim
in the board presentation, first agree on the dollar value of each penalty unit.

## Business case

- **Data engineering:** checksums and ledgers prevent duplicate files from corrupting reports; WAP audits isolate bad rows before Silver publication; BI and ML consumers share trusted Gold assets.
- **Data science:** the model must beat the free heuristic of using the same station and hour one week earlier.
- **Reliability:** Docker Compose makes the local environment reproducible and maintainable.
- **Outcome:** better station-demand decisions reduce missed rides, protect customer satisfaction, and support revenue.

## Architecture

```text
Official monthly ZIPs -> normalized Parquet -> manifest/asset event
  -> Bronze Iceberg raw table + checksum ledger
  -> Silver WAP candidate -> audit -> publish or quarantine
  -> Gold BI station operations + Gold ML station-hour demand
  -> leakage-safe, cost-sensitive ML training
  -> hourly prediction + last-week baseline
  -> Iceberg -> PostgreSQL -> FastAPI -> Streamlit
```

## Interfaces

| Service | URL | Login |
|---|---|---|
| Streamlit | http://localhost:8501 | none |
| Airflow | http://localhost:8080 | `admin` / `admin` |
| Jupyter | http://localhost:8888 | token `dev_token` |
| FastAPI docs | http://localhost:8000/docs | none |

## Run from a clean checkout

```bash
bash init.sh
docker compose up -d --build
```

In Airflow, unpause DAGs 1–3. Keep DAG 4 paused until training completes.

Download and convert all 17 official monthly archives (about 10.4 GiB compressed):

```bash
docker compose exec -T jupyter-notebook \
  python /opt/airflow/jobs/download_raw_data.py --start 2025-01 --end 2026-05
```

Publish the monthly manifests and allow Airflow to build Bronze/Silver/Gold:

```bash
bash run_all_months.sh
```

Audit and train:

```bash
docker compose exec -T jupyter-notebook python /opt/airflow/jobs/audit_gold_coverage.py
docker compose exec -T jupyter-notebook python /opt/airflow/jobs/ml_train_pipeline.py
```

Unpause DAG 4, then create historical predictions for May 2026:

```bash
bash backfill_x_days.sh
```

The backfill uses one Spark session for all 312 evaluation hours, then refreshes
PostgreSQL in one operation. This avoids launching hundreds of JVMs on a laptop.
The hourly DAG defaults to a three-month presentation shift, so an August 2026
clock produces May 2026 predictions. The historical bulk backfill deliberately
uses two months because its simulated clock is July. The data-availability guard
skips any target for which Gold actuals do not yet exist.

Verify:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/stations
```

## Data contract

Official fields retained in Bronze include ride ID/type, start/end timestamps, station names/IDs, coordinates, and member/casual status. Silver rejects missing IDs/timestamps/stations, invalid time order or duration, invalid coordinates/member type, source-month mismatch, and duplicate ride IDs. Audit results are stored in `nyc.audit.wap_audit_results`; rejected rows go to `nyc.quarantine.citibike_trips`.

Gold ML contains only station ID/name, station-hour timestamp, and ride count. Training reconstructs explicit zero-demand hours and uses chronological splits. All rolling features are shifted to prevent target leakage.

## Model and baseline

Demand is classified as Low/Normal/High using q25/q75 thresholds fitted on training data only. Optuna minimizes the asymmetric operational penalty matrix. Features include weekly/daily/hourly lags, rolling statistics, cyclical time, rush-hour indicators, demand velocity, acceleration, and short-versus-long trend gaps.

The business baseline is **the same station and same hour 168 hours earlier**, not the previous hour.

## Safe shutdown

```bash
docker compose down
```

This preserves data and database volumes. `docker compose down -v` deletes database state and should only be used for an intentional full reset.
