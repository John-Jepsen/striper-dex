#!/usr/bin/env python3
"""
Collect barometric pressure data from NOAA CO-OPS API.

Barometric pressure is a key indicator for fishing conditions:
- Falling pressure often indicates approaching storms and poor fishing
- Rising pressure suggests clearing weather and improved fishing
- Stable high pressure is generally favorable
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path
from typing import List, Tuple
import urllib.error
import urllib.request
import csv
import io

NOAA_API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"


def fetch_pressure_csv(
    station: str,
    start_date: dt.date,
    end_date: dt.date,
    *,
    application: str = "monterey-fishing-prediction",
    units: str = "english",
    time_zone: str = "lst_ldt",
    interval: str = "h",
    timeout: int = 60,
) -> str:
    """Fetch hourly barometric pressure data from NOAA CO-OPS."""
    params = (
        f"?product=air_pressure"
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


def parse_pressure_rows(csv_text: str) -> List[Tuple[dt.datetime, float]]:
    """Parse NOAA pressure CSV into list of (timestamp, pressure_mb) tuples."""
    reader = csv.DictReader(io.StringIO(csv_text))
    rows: List[Tuple[dt.datetime, float]] = []
    
    for row in reader:
        # Find datetime column (varies by API response)
        dt_key = next((k for k in row if "date" in k.lower() and "time" in k.lower()), None)
        # Find pressure column
        pressure_key = next((k for k in row if "pressure" in k.lower()), None)
        
        if not dt_key or not pressure_key:
            continue
            
        raw_dt = str(row.get(dt_key, "")).strip()
        raw_pressure = str(row.get(pressure_key, "")).strip()
        
        if not raw_dt or not raw_pressure:
            continue
            
        try:
            timestamp = dt.datetime.strptime(raw_dt, "%Y-%m-%d %H:%M")
        except ValueError:
            continue
            
        try:
            # NOAA returns millibars
            pressure_mb = float(raw_pressure)
        except ValueError:
            continue
            
        rows.append((timestamp, pressure_mb))
    
    return rows


def calculate_pressure_trend(readings: List[Tuple[dt.datetime, float]], window_hours: int = 3) -> List[Tuple[dt.datetime, float, float]]:
    """
    Calculate pressure change trend over rolling window.
    Returns: (timestamp, pressure_mb, change_mb_per_hour)
    """
    if len(readings) < 2:
        return []
    
    results: List[Tuple[dt.datetime, float, float]] = []
    
    for i in range(len(readings)):
        timestamp, pressure = readings[i]
        
        # Look back window_hours
        lookback_idx = max(0, i - window_hours)
        if lookback_idx == i:
            results.append((timestamp, pressure, 0.0))
            continue
            
        past_time, past_pressure = readings[lookback_idx]
        hours_diff = (timestamp - past_time).total_seconds() / 3600
        
        if hours_diff > 0:
            change_per_hour = (pressure - past_pressure) / hours_diff
        else:
            change_per_hour = 0.0
            
        results.append((timestamp, pressure, change_per_hour))
    
    return results


def save_pressure_csv(path: Path, data: List[Tuple[dt.datetime, float, float]]) -> None:
    """Save pressure data with trends to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'pressure_mb', 'pressure_change_mb_per_hour'])
        
        for timestamp, pressure, change in data:
            writer.writerow([
                timestamp.strftime("%Y-%m-%d %H:%M"),
                f"{pressure:.2f}",
                f"{change:.4f}"
            ])


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect barometric pressure data from NOAA.")
    parser.add_argument(
        "--station",
        default="9413450",
        help="NOAA CO-OPS station (default: 9413450 Monterey Harbor)"
    )
    parser.add_argument(
        "--start",
        type=lambda s: dt.datetime.strptime(s, "%Y-%m-%d").date(),
        default=dt.date.today() - dt.timedelta(days=365),
        help="Start date YYYY-MM-DD (default: 1 year ago)"
    )
    parser.add_argument(
        "--end",
        type=lambda s: dt.datetime.strptime(s, "%Y-%m-%d").date(),
        default=dt.date.today(),
        help="End date YYYY-MM-DD (default: today)"
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=30,
        help="Days per API request (default: 30)"
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory (default: data/processed)"
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.5,
        help="Seconds between API calls (default: 0.5)"
    )
    
    args = parser.parse_args(argv)
    
    print(f"Collecting barometric pressure for station {args.station}")
    print(f"Period: {args.start} to {args.end}")
    
    all_readings: List[Tuple[dt.datetime, float]] = []
    
    # Chunk requests to avoid API limits
    cursor = args.start
    while cursor <= args.end:
        chunk_end = min(cursor + dt.timedelta(days=args.chunk_days - 1), args.end)
        
        print(f"Fetching {cursor} to {chunk_end}...", end=" ", flush=True)
        
        try:
            csv_text = fetch_pressure_csv(args.station, cursor, chunk_end)
            chunk_readings = parse_pressure_rows(csv_text)
            all_readings.extend(chunk_readings)
            print(f"{len(chunk_readings)} readings")
        except urllib.error.URLError as e:
            print(f"\nError: {e}", file=sys.stderr)
            return 1
        
        cursor = chunk_end + dt.timedelta(days=1)
        
        if cursor <= args.end:
            time.sleep(args.pause)
    
    if not all_readings:
        print("No pressure data retrieved!", file=sys.stderr)
        return 1
    
    # Remove duplicates, sort by time
    unique_readings = sorted(set(all_readings), key=lambda x: x[0])
    
    print(f"\nCollected {len(unique_readings)} unique pressure readings")
    
    # Calculate trends
    with_trends = calculate_pressure_trend(unique_readings, window_hours=3)
    
    # Save to CSV
    outfile = args.outdir / f"{args.station}_barometric_pressure.csv"
    save_pressure_csv(outfile, with_trends)
    
    print(f"Saved to {outfile}")
    
    # Print summary statistics
    pressures = [p for _, p, _ in with_trends]
    changes = [c for _, _, c in with_trends]
    
    print(f"\nPressure Summary:")
    print(f"  Min: {min(pressures):.2f} mb")
    print(f"  Max: {max(pressures):.2f} mb")
    print(f"  Mean: {sum(pressures)/len(pressures):.2f} mb")
    print(f"\nPressure Change Summary:")
    print(f"  Max rise: {max(changes):.4f} mb/hr")
    print(f"  Max fall: {min(changes):.4f} mb/hr")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
