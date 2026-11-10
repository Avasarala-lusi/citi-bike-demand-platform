#!/usr/bin/env python3
"""Download official Citi Bike monthly archives and convert every CSV part to Parquet."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import sys
import zipfile

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import requests


SOURCE_COLUMNS = [
    "ride_id",
    "rideable_type",
    "started_at",
    "ended_at",
    "start_station_name",
    "start_station_id",
    "end_station_name",
    "end_station_id",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
    "member_casual",
]
STRING_SCHEMA = pa.schema([pa.field(name, pa.string()) for name in SOURCE_COLUMNS])


def month_range(start: str, end: str):
    start_year, start_month = map(int, start.split("-"))
    end_year, end_month = map(int, end.split("-"))
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        yield year, month
        month += 1
        if month == 13:
            year += 1
            month = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Skipping existing archive: {destination.name}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        expected = int(response.headers.get("content-length", 0))
        written = 0
        with partial.open("wb") as output:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    output.write(chunk)
                    written += len(chunk)
                    if expected:
                        print(
                            f"  {destination.name}: {written / expected:6.1%}",
                            end="\r",
                            flush=True,
                        )
    partial.replace(destination)
    print(f"  {destination.name}: complete ({destination.stat().st_size:,} bytes)")


def normalize_batch(batch: pa.RecordBatch) -> pa.Table:
    table = pa.Table.from_batches([batch])
    arrays = []
    for name in SOURCE_COLUMNS:
        if name in table.column_names:
            arrays.append(table[name].cast(pa.string()))
        else:
            arrays.append(pa.nulls(table.num_rows, type=pa.string()))
    return pa.Table.from_arrays(arrays, schema=STRING_SCHEMA)


def archive_to_parquet(archive: Path, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Skipping existing Parquet: {destination.name}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    writer = pq.ParquetWriter(partial, STRING_SCHEMA, compression="snappy")
    row_count = 0
    csv_count = 0

    try:
        with zipfile.ZipFile(archive) as bundle:
            members = sorted(
                member
                for member in bundle.namelist()
                if member.lower().endswith(".csv") and not member.startswith("__MACOSX/")
            )
            if not members:
                raise ValueError(f"No CSV files found in {archive}")

            for member in members:
                csv_count += 1
                print(f"  Converting {member}")
                with bundle.open(member) as source:
                    reader = pacsv.open_csv(
                        source,
                        read_options=pacsv.ReadOptions(block_size=32 * 1024 * 1024),
                        convert_options=pacsv.ConvertOptions(
                            column_types={name: pa.string() for name in SOURCE_COLUMNS},
                            strings_can_be_null=True,
                        ),
                    )
                    for batch in reader:
                        table = normalize_batch(batch)
                        writer.write_table(table)
                        row_count += table.num_rows
    finally:
        writer.close()

    if row_count == 0:
        partial.unlink(missing_ok=True)
        raise ValueError(f"Converted archive contained zero rows: {archive}")
    partial.replace(destination)
    print(
        f"Created {destination.name}: {row_count:,} rides from {csv_count} CSV part(s); "
        f"archive_sha256={sha256(archive)}"
    )


def process_month(year: int, month: int, root: Path) -> None:
    key = f"{year}{month:02d}"
    archive_name = f"{key}-citibike-tripdata.zip"
    parquet_name = f"citibike_tripdata_{year}-{month:02d}.parquet"
    archive = root / "source_archives" / archive_name
    parquet = root / "incoming" / parquet_name
    url = f"https://s3.amazonaws.com/tripdata/{archive_name}"
    download(url, archive)
    archive_to_parquet(archive, parquet)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-01", help="First month, YYYY-MM")
    parser.add_argument("--end", default="2026-05", help="Last month, YYYY-MM")
    parser.add_argument("--output-root", default="/opt/airflow/data")
    args = parser.parse_args()

    root = Path(args.output_root)
    for year, month in month_range(args.start, args.end):
        try:
            process_month(year, month, root)
        except Exception as exc:
            print(f"FAILED {year}-{month:02d}: {exc}", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
