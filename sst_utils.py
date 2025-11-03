#!/usr/bin/env python3
"""Utility helpers for working with NOAA SST data."""

from __future__ import annotations

import csv
import datetime as dt
import io
import statistics as stats
from typing import Iterable, List, Sequence, Tuple
import urllib.error
import urllib.request

NOAA_API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"


def fetch_hourly_csv(
    station: str,
    start_date: dt.date,
    end_date: dt.date,
    *,
    application: str = "monterey-weekly-sst",
    units: str = "english",
    time_zone: str = "lst_ldt",
    interval: str = "h",
    timeout: int = 60,
) -> str:
    params = (
        f"?product=water_temperature"
        f"&application={application}"
        f"&begin_date={start_date:%Y%m%d}"
        f"&end_date={end_date:%Y%m%d}"
        f"&station={station}"
        f"&time_zone={time_zone}"
        f"&units={units}"
        f"&interval={interval}"
        f"&format=csv"
    )
    url = NOAA_API + params
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_hourly_rows(csv_text: str) -> List[Tuple[dt.datetime, float]]:
    reader: Iterable[dict[str, str]] = csv.DictReader(io.StringIO(csv_text))
    rows: List[Tuple[dt.datetime, float]] = []
    for row in reader:
        dt_key = next((k for k in row if k.strip().lower().startswith("date time")), None)
        temp_key = next((k for k in row if "water temperature" in k.lower()), None)
        if not dt_key or not temp_key:
            continue
        raw_dt_val = row.get(dt_key, "")
        raw_temp_val = row.get(temp_key, "")
        raw_dt = raw_dt_val.strip() if isinstance(raw_dt_val, str) else str(raw_dt_val).strip()
        raw_temp = raw_temp_val.strip() if isinstance(raw_temp_val, str) else str(raw_temp_val).strip()
        if not raw_dt:
            continue
        try:
            timestamp = dt.datetime.strptime(raw_dt, "%Y-%m-%d %H:%M")
        except ValueError:
            continue
        if not raw_temp:
            continue
        try:
            temp_f = float(raw_temp)
        except ValueError:
            continue
        rows.append((timestamp, temp_f))
    return rows


def morning_snapshots(
    hourly: Sequence[Tuple[dt.datetime, float]],
    *,
    target_hour: int = 7,
    window: Tuple[int, int] = (6, 9),
) -> List[Tuple[dt.date, float]]:
    by_day: dict[dt.date, List[Tuple[dt.datetime, float]]] = {}
    for ts, temp_f in hourly:
        if window[0] <= ts.hour <= window[1]:
            by_day.setdefault(ts.date(), []).append((ts, temp_f))
    snapshots: List[Tuple[dt.date, float]] = []
    for date_key, values in by_day.items():
        pick_ts, pick_temp = min(values, key=lambda entry: abs(entry[0].hour - target_hour))
        snapshots.append((pick_ts.date(), pick_temp))
    return sorted(snapshots, key=lambda entry: entry[0])


def iso_year_week(d: dt.date) -> Tuple[int, int]:
    year, week, _ = d.isocalendar()
    return int(year), int(week)


def weekly_aggregate(
    snapshots: Sequence[Tuple[dt.date, float]]
) -> List[dict[str, object]]:
    buckets: dict[Tuple[int, int], List[float]] = {}
    for date_key, temp_f in snapshots:
        buckets.setdefault(iso_year_week(date_key), []).append(temp_f)
    weekly_rows: List[dict[str, object]] = []
    for year_week in sorted(buckets):
        year, week = year_week
        readings = buckets[year_week]
        week_start = dt.date.fromisocalendar(year, week, 1)
        week_end = dt.date.fromisocalendar(year, week, 7)
        weekly_rows.append(
            {
                "iso_year": year,
                "iso_week": week,
                "week_start": week_start.isoformat(),
                "week_end": week_end.isoformat(),
                "days_count": len(readings),
                "morning_temp_avg_F": round(stats.fmean(readings), 2),
            }
        )
    return weekly_rows


def write_csv(path: str, rows: Sequence[dict[str, object]]) -> None:
    fieldnames = [
        "iso_year",
        "iso_week",
        "week_start",
        "week_end",
        "days_count",
        "morning_temp_avg_F",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


__all__ = [
    "NOAA_API",
    "fetch_hourly_csv",
    "iso_year_week",
    "morning_snapshots",
    "parse_hourly_rows",
    "weekly_aggregate",
    "write_csv",
]
