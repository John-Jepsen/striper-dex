#!/usr/bin/env python3
"""
Weekly morning water temperature averages for Monterey, CA (default station 9413450).
- Pulls NOAA CO-OPS hourly temps for the last N days (local time, LST/LDT).
- Picks one morning snapshot per day (prefer 07:00; else nearest in 06–09).
- Buckets by ISO week and writes a CSV.
"""

import argparse
import datetime as dt
import sys
import urllib.error

from sst_utils import (
    fetch_hourly_csv,
    morning_snapshots,
    parse_hourly_rows,
    weekly_aggregate,
    write_csv,
)

def main():
    ap = argparse.ArgumentParser(description="Weekly morning SST averages from NOAA CO-OPS.")
    ap.add_argument("--station", default="9413450", help="NOAA CO-OPS station (default: 9413450 Monterey Harbor)")
    ap.add_argument("--days", type=int, default=350, help="Lookback days (default: 150)")
    ap.add_argument("--outfile", default="monterey_morning_temps_weekly_last150days.csv", help="Output CSV path")
    ap.add_argument("--target-hour", type=int, default=7, help="Preferred morning hour (default: 7)")
    ap.add_argument("--window", default="6-9", help="Morning hour window inclusive, e.g., 6-9")
    args = ap.parse_args()

    try:
        w_start, w_end = map(int, args.window.split("-"))
        if not (0 <= w_start <= 23 and 0 <= w_end <= 23 and w_start <= w_end):
            raise ValueError
    except Exception:
        print("Invalid --window; use format H1-H2 within 0–23, e.g., 6-9", file=sys.stderr)
        sys.exit(2)

    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=args.days)

    try:
        text = fetch_hourly_csv(args.station, start_date, end_date)
    except urllib.error.URLError as e:
        print(f"Fetch error: {e}", file=sys.stderr)
        sys.exit(1)

    hourly = parse_hourly_rows(text)
    if not hourly:
        print("No valid readings found in response.", file=sys.stderr)
        sys.exit(1)

    shots = morning_snapshots(hourly, target_hour=args.target_hour, window=(w_start, w_end))
    # Keep at most N distinct days, most recent first
    shots = shots[-args.days:]

    weekly = weekly_aggregate(shots)
    write_csv(args.outfile, weekly)

    # Pretty print summary to stdout
    print(f"Wrote {len(weekly)} weekly rows → {args.outfile}")
    for r in weekly[-6:]:
        print(f"{r['week_start']}..{r['week_end']}  "
              f"days:{r['days_count']:>2}  avg_F:{r['morning_temp_avg_F']:.2f}")

if __name__ == "__main__":
    main()