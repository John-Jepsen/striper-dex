#!/usr/bin/env python3
"""Backfill full NOAA CO-OPS water temperature history for a station."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple
import urllib.error

from sst_utils import (
    fetch_hourly_csv,
    morning_snapshots,
    parse_hourly_rows,
    weekly_aggregate,
    write_csv,
)


def parse_date(value: str) -> dt.date:
    try:
        return dt.datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date '{value}', expected YYYY-MM-DD") from exc


def parse_window(value: str) -> Tuple[int, int]:
    try:
        start_str, end_str = value.split("-", 1)
        start_hour = int(start_str)
        end_hour = int(end_str)
    except Exception as exc:
        raise argparse.ArgumentTypeError("Window must look like 6-9") from exc
    if start_hour < 0 or end_hour > 23 or start_hour > end_hour:
        raise argparse.ArgumentTypeError("Window hours must be between 0 and 23 inclusive")
    return start_hour, end_hour


def iter_chunks(
    start: dt.date,
    end: dt.date,
    chunk_days: int,
) -> Iterator[Tuple[dt.date, dt.date]]:
    cursor = start
    one_day = dt.timedelta(days=1)
    max_delta = dt.timedelta(days=chunk_days - 1)
    while cursor <= end:
        chunk_end = min(cursor + max_delta, end)
        yield cursor, chunk_end
        cursor = chunk_end + one_day


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_hourly_for_range(
    station: str,
    chunk_start: dt.date,
    chunk_end: dt.date,
    raw_dir: Path,
    *,
    force: bool = False,
    pause_seconds: float = 0.0,
) -> List[Tuple[dt.datetime, float]]:
    raw_dir = raw_dir.resolve()
    ensure_dir(raw_dir)
    filename = f"{station}_{chunk_start:%Y%m%d}_{chunk_end:%Y%m%d}.csv"
    raw_path = raw_dir / filename
    text: str
    if raw_path.exists() and not force:
        text = raw_path.read_text()
        source = "cache"
    else:
        try:
            text = fetch_hourly_csv(station, chunk_start, chunk_end)
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to fetch {chunk_start:%Y-%m-%d} to {chunk_end:%Y-%m-%d}: {exc}") from exc
        raw_path.write_text(text)
        source = "download"
        if pause_seconds > 0:
            time.sleep(pause_seconds)
    hourly = parse_hourly_rows(text)
    print(
        f"{chunk_start:%Y-%m-%d} → {chunk_end:%Y-%m-%d}: {len(hourly)} readings ({source})",
        flush=True,
    )
    return hourly


def dedupe_hourly(
    readings: Iterable[Tuple[dt.datetime, float]]
) -> List[Tuple[dt.datetime, float]]:
    latest: dict[dt.datetime, float] = {}
    for timestamp, temp_f in readings:
        latest[timestamp] = temp_f
    return sorted(latest.items(), key=lambda entry: entry[0])


def write_daily_csv(path: Path, snapshots: Sequence[Tuple[dt.date, float]]) -> None:
    ensure_dir(path.parent)
    lines = ["date,morning_temp_F\n"]
    for date_key, temp_f in snapshots:
        lines.append(f"{date_key.isoformat()},{temp_f:.2f}\n")
    path.write_text("".join(lines))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download chunked NOAA CO-OPS temperature history and aggregate by week.",
    )
    parser.add_argument("--station", default="9413450", help="CO-OPS station id (default: 9413450 Monterey Harbor)")
    parser.add_argument(
        "--start",
        type=parse_date,
        default=dt.date(1990, 1, 1),
        help="Earliest date to request (default: 1990-01-01)",
    )
    parser.add_argument(
        "--end",
        type=parse_date,
        default=dt.date.today(),
        help="Last date to request (default: today)",
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=365,
        help="Number of days per request (max NOAA allows 365)",
    )
    parser.add_argument(
        "--target-hour",
        type=int,
        default=7,
        help="Preferred morning hour for snapshots (default: 7)",
    )
    parser.add_argument(
        "--window",
        type=parse_window,
        default=(6, 9),
        help="Morning hour window inclusive, e.g., 6-9 (default: 6-9)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/noaa"),
        help="Directory to store raw chunk CSVs (default: data/raw/noaa)",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for processed outputs (default: data/processed)",
    )
    parser.add_argument(
        "--weekly-out",
        type=Path,
        default=None,
        help="Override processed weekly CSV path (default is derived from station)",
    )
    parser.add_argument(
        "--daily-out",
        type=Path,
        default=None,
        help="Optional daily snapshot CSV output path",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.5,
        help="Seconds to pause between downloads (default: 0.5)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download chunks even if cached")

    args = parser.parse_args(argv)

    if args.chunk_days < 1 or args.chunk_days > 366:
        parser.error("--chunk-days must be between 1 and 366")
    if args.end < args.start:
        parser.error("--end must be on or after --start")
    window_start, window_end = args.window

    raw_dir = args.raw_dir
    processed_dir = args.processed_dir
    ensure_dir(raw_dir)
    ensure_dir(processed_dir)

    weekly_path = (
        args.weekly_out
        if args.weekly_out is not None
        else processed_dir / f"{args.station}_weekly_morning_temps.csv"
    )
    daily_path = args.daily_out or processed_dir / f"{args.station}_morning_daily.csv"

    print(
        f"Pulling station {args.station} from {args.start:%Y-%m-%d} to {args.end:%Y-%m-%d} "
        f"in {args.chunk_days}-day chunks...",
        flush=True,
    )

    all_hourly: List[Tuple[dt.datetime, float]] = []
    chunk_iter = list(iter_chunks(args.start, args.end, args.chunk_days))
    total_chunks = len(chunk_iter)
    for idx, (chunk_start, chunk_end) in enumerate(chunk_iter, start=1):
        print(f"[{idx}/{total_chunks}]", end=" ", flush=True)
        try:
            hourly = load_hourly_for_range(
                args.station,
                chunk_start,
                chunk_end,
                raw_dir,
                force=args.force,
                pause_seconds=args.pause,
            )
        except RuntimeError as exc:
            print(f"\n{exc}", file=sys.stderr)
            return 1
        all_hourly.extend(hourly)

    if not all_hourly:
        print("No hourly readings retrieved.", file=sys.stderr)
        return 1

    deduped = dedupe_hourly(all_hourly)
    print(f"Collected {len(deduped)} unique hourly readings total.")

    snapshots = morning_snapshots(
        deduped,
        target_hour=args.target_hour,
        window=(window_start, window_end),
    )
    print(f"Derived {len(snapshots)} morning snapshots after filtering.")

    write_daily_csv(daily_path, snapshots)
    print(f"Daily snapshots written to {daily_path}")

    weekly_rows = weekly_aggregate(snapshots)
    write_csv(str(weekly_path), weekly_rows)
    print(f"Weekly aggregates written to {weekly_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
