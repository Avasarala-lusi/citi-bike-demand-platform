from contextlib import asynccontextmanager
from datetime import datetime
import os

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool


PG_HOST = os.getenv("POSTGRES_CACHE_HOST", "operational_cache_postgres")
PG_PORT = int(os.getenv("POSTGRES_CACHE_PORT", "5432"))
PG_DB = os.getenv("POSTGRES_CACHE_DB", "operational_cache")
PG_USER = os.getenv("POSTGRES_CACHE_USER", "cache_user")
PG_PASSWORD = os.getenv("POSTGRES_CACHE_PASSWORD", "cache_password")
db_container = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db_container["pool"] = ThreadedConnectionPool(
        minconn=1, maxconn=10, host=PG_HOST, port=PG_PORT,
        dbname=PG_DB, user=PG_USER, password=PG_PASSWORD,
    )
    try:
        yield
    finally:
        db_container["pool"].closeall()


app = FastAPI(
    title="Citi Bike Demand Prediction Operational API",
    description="Serves cost-sensitive station demand predictions from PostgreSQL.",
    lifespan=lifespan,
)


def get_db():
    pool = db_container.get("pool")
    if pool is None:
        raise HTTPException(status_code=503, detail="Database pool is not ready")
    connection = pool.getconn()
    try:
        yield connection
    finally:
        pool.putconn(connection)


def clean_record(record: dict) -> dict:
    cleaned = {}
    for key, value in record.items():
        if pd.isna(value):
            cleaned[key] = None
        elif isinstance(value, (pd.Timestamp, datetime)):
            cleaned[key] = value.isoformat()
        else:
            cleaned[key] = value
    return cleaned


def predictions_table_exists(connection) -> bool:
    with connection.cursor() as cursor:
        cursor.execute("SELECT to_regclass('public.hourly_demand_predictions') IS NOT NULL")
        return bool(cursor.fetchone()[0])


@app.get("/health")
def health(db=Depends(get_db)):
    with db.cursor() as cursor:
        cursor.execute("SELECT 1")
    return {
        "status": "ok",
        "database_connected": True,
        "predictions_ready": predictions_table_exists(db),
        "database_host": PG_HOST,
    }


@app.get("/stations")
def stations(db=Depends(get_db)):
    if not predictions_table_exists(db):
        return {"stations": []}
    with db.cursor() as cursor:
        cursor.execute("SELECT DISTINCT station_id FROM hourly_demand_predictions ORDER BY station_id")
        values = [row[0] for row in cursor.fetchall()]
    return {"stations": values}


@app.get("/predictions/{station_id}/{prediction_hour_ts}")
def get_prediction(station_id: str, prediction_hour_ts: str, db=Depends(get_db)):
    if not predictions_table_exists(db):
        raise HTTPException(status_code=503, detail="Scoring has not created the predictions table yet")
    try:
        timestamp = pd.Timestamp(prediction_hour_ts).floor("h").to_pydatetime()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid timestamp") from exc
    with db.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute(
            "SELECT * FROM hourly_demand_predictions WHERE station_id=%s AND prediction_hour_ts=%s",
            (station_id, timestamp),
        )
        row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return clean_record(dict(row))


@app.get("/predictions/{station_id}")
def list_predictions(station_id: str, db=Depends(get_db)):
    if not predictions_table_exists(db):
        return {"station_id": station_id, "count": 0, "predictions": []}
    try:
        with db.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                "SELECT * FROM hourly_demand_predictions WHERE station_id=%s ORDER BY prediction_hour_ts",
                (station_id,),
            )
            rows = cursor.fetchall()
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {exc}") from exc
    return {
        "station_id": station_id,
        "count": len(rows),
        "predictions": [clean_record(dict(row)) for row in rows],
    }
