#!/usr/bin/env python3
"""
Collect tidal (water level) data from NOAA CO-OPS API.

Tidal phase is critical for striped bass fishing:
- Flood tide (incoming): Fish move into bays/shallows to feed
- Ebb tide (outgoing): Fish follow currents to inlets/river mouths
- Slack tide: Reduced activity
- Moving water: Most productive fishing

API Documentation: https://api.tidesandcurrents.noaa.gov/api/prod/
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import requests


def fetch_tidal_data(
    station: str,
    start_date: str,
    end_date: str,
    product: str = "water_level"
) -> pd.DataFrame:
    """
    Fetch tidal data from NOAA CO-OPS API.
    
    Product options:
    - water_level: Verified 6-minute water levels
    - predictions: Tide predictions (for future forecasts)
    """
    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    
    params = {
        "product": product,
        "station": station,
        "begin_date": start_date.replace("-", ""),
        "end_date": end_date.replace("-", ""),
        "datum": "MLLW",  # Mean Lower Low Water
        "time_zone": "GMT",
        "units": "english",
        "format": "json",
        "application": "fishing_predictions"
    }
    
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    
    data = response.json()
    
    if "data" not in data:
        raise ValueError(f"No data returned: {data.get('error', {}).get('message', 'Unknown error')}")
    
    df = pd.DataFrame(data["data"])
    df["t"] = pd.to_datetime(df["t"])
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    
    df.rename(columns={"t": "timestamp", "v": "water_level_ft"}, inplace=True)
    
    return df[["timestamp", "water_level_ft"]]


def calculate_tidal_phase(df: pd.DataFrame, window_hours: int = 3) -> pd.DataFrame:
    """
    Calculate tidal phase from water level data.
    
    Phases:
    - flood: Water rising (incoming tide)
    - ebb: Water falling (outgoing tide)  
    - slack_high: Near high tide with minimal movement
    - slack_low: Near low tide with minimal movement
    """
    df = df.copy()
    
    # Calculate rate of change (ft/hour)
    df["tidal_rate_ft_per_hr"] = df["water_level_ft"].diff() / (
        df["timestamp"].diff().dt.total_seconds() / 3600
    )
    
    # Smooth the rate to reduce noise
    df["tidal_rate_smooth"] = df["tidal_rate_ft_per_hr"].rolling(
        window=window_hours, center=True, min_periods=1
    ).mean()
    
    # Classify phase based on rate
    def classify_phase(rate):
        if pd.isna(rate):
            return "unknown"
        elif rate > 0.1:
            return "flood"
        elif rate < -0.1:
            return "ebb"
        elif abs(rate) <= 0.1:
            # Check if we're near high or low (needs context)
            return "slack"
        else:
            return "unknown"
    
    df["tidal_phase"] = df["tidal_rate_smooth"].apply(classify_phase)
    
    # Identify high/low tides (local extrema)
    df["is_high_tide"] = (
        (df["water_level_ft"] > df["water_level_ft"].shift(1)) &
        (df["water_level_ft"] > df["water_level_ft"].shift(-1))
    ).astype(int)
    
    df["is_low_tide"] = (
        (df["water_level_ft"] < df["water_level_ft"].shift(1)) &
        (df["water_level_ft"] < df["water_level_ft"].shift(-1))
    ).astype(int)
    
    # Refine slack classification
    df.loc[(df["tidal_phase"] == "slack") & (df["is_high_tide"] == 1), "tidal_phase"] = "slack_high"
    df.loc[(df["tidal_phase"] == "slack") & (df["is_low_tide"] == 1), "tidal_phase"] = "slack_low"
    
    # Calculate tidal range (high to low difference)
    high_tides = df[df["is_high_tide"] == 1]["water_level_ft"]
    low_tides = df[df["is_low_tide"] == 1]["water_level_ft"]
    
    if len(high_tides) > 0 and len(low_tides) > 0:
        tidal_range = high_tides.mean() - low_tides.mean()
        df["tidal_range_ft"] = tidal_range
    else:
        df["tidal_range_ft"] = None
    
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect tidal data from NOAA.")
    parser.add_argument(
        "--station",
        default="9413450",
        help="NOAA station ID (default: 9413450 Monterey Harbor)"
    )
    parser.add_argument(
        "--start",
        default=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
        help="Start date YYYY-MM-DD (default: 1 year ago)"
    )
    parser.add_argument(
        "--end",
        default=datetime.now().strftime("%Y-%m-%d"),
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
    
    # Parse dates
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    
    print(f"Collecting tidal data for station {args.station}")
    print(f"Date range: {args.start} to {args.end}")
    
    # Fetch data in chunks
    all_data = []
    current = start
    
    while current < end:
        chunk_end = min(current + timedelta(days=args.chunk_days), end)
        
        start_str = current.strftime("%Y-%m-%d")
        end_str = chunk_end.strftime("%Y-%m-%d")
        
        try:
            print(f"Fetching {start_str} to {end_str}...", end=" ")
            df = fetch_tidal_data(args.station, start_str, end_str)
            all_data.append(df)
            print(f"{len(df)} readings")
            
        except Exception as e:
            print(f"Error: {e}")
        
        current = chunk_end
        time.sleep(args.pause)
    
    # Combine all data
    if not all_data:
        print("No data collected!")
        return 1
    
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    
    print(f"\nCollected {len(combined)} unique tidal readings")
    
    # Calculate tidal phases
    print("Calculating tidal phases...")
    combined = calculate_tidal_phase(combined)
    
    # Save to CSV
    args.outdir.mkdir(parents=True, exist_ok=True)
    output_file = args.outdir / f"{args.station}_tidal_data.csv"
    combined.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")
    
    # Print summary
    print("\nTidal Summary:")
    print(f"  Water level range: {combined['water_level_ft'].min():.2f} - {combined['water_level_ft'].max():.2f} ft")
    if combined["tidal_range_ft"].notna().any():
        print(f"  Average tidal range: {combined['tidal_range_ft'].iloc[0]:.2f} ft")
    
    print("\nPhase distribution:")
    print(combined["tidal_phase"].value_counts())
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
