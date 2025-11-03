#!/usr/bin/env python3
"""
Complete pipeline runner for enhanced fishing forecast.

Orchestrates:
1. Weather data collection (NDBC)
2. Tidal data collection (NOAA CO-OPS)
3. Barometric pressure collection
4. Feature engineering with all data sources
5. Enhanced forecast generation with validation

Run this weekly to keep forecasts fresh.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and report success/failure."""
    print(f"\n{'='*70}")
    print(f"📍 {description}")
    print(f"{'='*70}")
    print(f"Running: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} - FAILED")
        print(f"Error: {e}")
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run complete fishing forecast pipeline."
    )
    parser.add_argument(
        "--skip-weather",
        action="store_true",
        help="Skip weather data collection (use existing)"
    )
    parser.add_argument(
        "--skip-tidal",
        action="store_true",
        help="Skip tidal data collection (use existing)"
    )
    parser.add_argument(
        "--skip-pressure",
        action="store_true",
        help="Skip pressure data collection (use existing)"
    )
    parser.add_argument(
        "--forecast-days",
        type=int,
        default=7,
        help="Number of days to forecast (default: 7)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional: Save forecast to CSV"
    )
    
    args = parser.parse_args(argv)
    
    print("🎣 ENHANCED FISHING FORECAST PIPELINE")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    
    success_count = 0
    total_steps = 0
    
    # Step 1: Collect weather data
    if not args.skip_weather:
        total_steps += 1
        if run_command(
            ["python", "collect_weather_data.py"],
            "Collect NDBC weather data (wind, upwelling, air temp)"
        ):
            success_count += 1
    else:
        print("\n⏭️  Skipping weather data collection")
    
    # Step 2: Collect tidal data
    if not args.skip_tidal:
        total_steps += 1
        # Only collect last 30 days to avoid long runtime
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        if run_command(
            ["python", "collect_tidal_data.py", "--start", start_date],
            "Collect tidal data (water levels, phases)"
        ):
            success_count += 1
    else:
        print("\n⏭️  Skipping tidal data collection")
    
    # Step 3: Collect barometric pressure
    if not args.skip_pressure:
        total_steps += 1
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        if run_command(
            ["python", "collect_barometric_pressure.py", "--start", start_date],
            "Collect barometric pressure data"
        ):
            success_count += 1
    else:
        print("\n⏭️  Skipping pressure data collection")
    
    # Step 4: Feature engineering
    total_steps += 1
    if run_command(
        [
            "python", "feature_engineering.py",
            "--weather-file", "data/processed/46042_weather_data.csv",
            "--tidal-file", "data/processed/9413450_tidal_data.csv"
        ],
        "Engineer features from all data sources"
    ):
        success_count += 1
    
    # Step 5: Generate forecast
    total_steps += 1
    cmd = ["python", "fishing_forecast.py", "--forecast-days", str(args.forecast_days)]
    if args.output:
        cmd.extend(["--output", str(args.output)])
    
    if run_command(
        cmd,
        f"Generate {args.forecast_days}-day forecast with validation"
    ):
        success_count += 1
    
    # Summary
    print("\n" + "="*70)
    print("📊 PIPELINE SUMMARY")
    print("="*70)
    print(f"Completed: {success_count}/{total_steps} steps")
    print(f"Finished: {datetime.now():%Y-%m-%d %H:%M:%S}")
    
    if success_count == total_steps:
        print("\n✅ All steps completed successfully!")
        print("\n🎣 Your enhanced fishing forecast is ready!")
        return 0
    else:
        print(f"\n⚠️  {total_steps - success_count} step(s) failed")
        print("Check output above for errors")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
