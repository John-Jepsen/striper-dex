#!/usr/bin/env python3
"""
Collect meteorological data from NOAA NDBC buoys.

Weather variables affect ocean temperature through:
- Wind speed/direction: Drives upwelling/downwelling
- Air temperature: Heat exchange with ocean surface
- Solar radiation: Primary heat input
- Barometric pressure: Storm systems affect mixing

NDBC Station 46042 (Monterey Bay) - 36.75°N 122.42°W
API: https://www.ndbc.noaa.gov/data/realtime2/
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import requests
from typing import Optional


def fetch_ndbc_realtime(station: str, timeout: int = 30) -> pd.DataFrame:
    """Fetch last 45 days of realtime data from NDBC."""
    url = f"https://www.ndbc.noaa.gov/data/realtime2/{station}.txt"
    
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch NDBC data: {e}")
    
    lines = response.text.strip().split('\n')
    if len(lines) < 3:
        raise ValueError("Insufficient data returned from NDBC")
    
    # NDBC format: header line 1 (names), line 2 (units), then data
    header = lines[0].split()
    units = lines[1].split()
    data_lines = lines[2:]
    
    records = []
    for line in data_lines:
        parts = line.split()
        if len(parts) < len(header):
            continue
        
        try:
            # Parse timestamp
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            hour = int(parts[3])
            minute = int(parts[4])
            timestamp = datetime(year, month, day, hour, minute)
            
            # Extract meteorological variables
            record = {'timestamp': timestamp}
            
            for i, name in enumerate(header):
                if i >= len(parts):
                    break
                value = parts[i]
                
                # Convert to numeric, handle NDBC's missing value codes (99, 999, MM)
                if value in ['MM', '999', '9999', '999.0', '99.0']:
                    record[name] = None
                else:
                    try:
                        record[name] = float(value)
                    except ValueError:
                        record[name] = None
            
            records.append(record)
            
        except (ValueError, IndexError):
            continue
    
    if not records:
        raise ValueError("No valid records parsed from NDBC data")
    
    df = pd.DataFrame(records)
    return df


def fetch_ndbc_historical_month(station: str, year: int, month: int, timeout: int = 30) -> pd.DataFrame:
    """Fetch historical monthly data from NDBC."""
    url = f"https://www.ndbc.noaa.gov/view_text_file.php?filename={station}h{year}{month:02d}.txt.gz&dir=data/historical/stdmet/"
    
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException:
        return pd.DataFrame()  # Month not available
    
    lines = response.text.strip().split('\n')
    if len(lines) < 3:
        return pd.DataFrame()
    
    header = lines[0].split()
    data_lines = lines[2:]
    
    records = []
    for line in data_lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        
        try:
            # Year might be 2-digit or 4-digit
            year_val = int(parts[0])
            if year_val < 100:
                year_val += 2000 if year_val < 50 else 1900
            
            month_val = int(parts[1])
            day = int(parts[2])
            hour = int(parts[3])
            minute = int(parts[4])
            timestamp = datetime(year_val, month_val, day, hour, minute)
            
            record = {'timestamp': timestamp}
            
            for i, name in enumerate(header):
                if i >= len(parts):
                    break
                value = parts[i]
                
                if value in ['MM', '999', '9999', '999.0', '99.0']:
                    record[name] = None
                else:
                    try:
                        record[name] = float(value)
                    except ValueError:
                        record[name] = None
            
            records.append(record)
            
        except (ValueError, IndexError):
            continue
    
    return pd.DataFrame(records)


def process_ndbc_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and process NDBC meteorological data."""
    df = df.copy()
    
    # Rename columns to be more descriptive
    column_map = {
        'WDIR': 'wind_direction_deg',
        'WSPD': 'wind_speed_mps',
        'GST': 'wind_gust_mps',
        'WVHT': 'wave_height_m',
        'DPD': 'wave_period_sec',
        'APD': 'wave_period_avg_sec',
        'MWD': 'wave_direction_deg',
        'PRES': 'pressure_hpa',
        'ATMP': 'air_temp_c',
        'WTMP': 'water_temp_c',
        'DEWP': 'dewpoint_c',
        'VIS': 'visibility_nmi',
        'TIDE': 'tide_ft',
    }
    
    df = df.rename(columns=column_map)
    
    # Convert to more common units
    if 'wind_speed_mps' in df.columns:
        df['wind_speed_kt'] = df['wind_speed_mps'] * 1.94384  # m/s to knots
        df['wind_speed_mph'] = df['wind_speed_mps'] * 2.23694
    
    if 'wind_gust_mps' in df.columns:
        df['wind_gust_kt'] = df['wind_gust_mps'] * 1.94384
    
    if 'air_temp_c' in df.columns:
        df['air_temp_f'] = df['air_temp_c'] * 9/5 + 32
    
    if 'water_temp_c' in df.columns:
        df['water_temp_f'] = df['water_temp_c'] * 9/5 + 32
    
    if 'pressure_hpa' in df.columns:
        df['pressure_mb'] = df['pressure_hpa']  # hPa = mb
        df['pressure_inhg'] = df['pressure_hpa'] * 0.02953
    
    # Calculate wind chill (simple approximation)
    if 'air_temp_f' in df.columns and 'wind_speed_mph' in df.columns:
        temp_f = df['air_temp_f']
        wind_mph = df['wind_speed_mph']
        
        # Wind chill formula (valid for T ≤ 50°F and wind > 3 mph)
        mask = (temp_f <= 50) & (wind_mph > 3)
        df['wind_chill_f'] = temp_f.copy()
        df.loc[mask, 'wind_chill_f'] = (
            35.74 + 
            0.6215 * temp_f[mask] - 
            35.75 * (wind_mph[mask] ** 0.16) + 
            0.4275 * temp_f[mask] * (wind_mph[mask] ** 0.16)
        )
    
    # Calculate upwelling-favorable wind (northward wind component)
    # For Monterey Bay, northerly winds (from north) cause upwelling
    if 'wind_direction_deg' in df.columns and 'wind_speed_kt' in df.columns:
        import numpy as np
        
        # Convert wind direction to radians (meteorological: direction FROM which wind blows)
        wind_dir_rad = np.radians(df['wind_direction_deg'])
        
        # Calculate northward component (positive = from south, negative = from north)
        # Upwelling-favorable = negative (from north)
        df['wind_north_component_kt'] = -df['wind_speed_kt'] * np.cos(wind_dir_rad)
        
        # Classify upwelling favorability
        df['upwelling_favorable'] = (df['wind_north_component_kt'] < -5).astype(int)
    
    # Keep only useful columns
    keep_cols = ['timestamp'] + [
        col for col in df.columns 
        if col not in ['YY', 'MM', 'DD', 'hh', 'mm', '#YY']
        and col != 'timestamp'
    ]
    
    df = df[keep_cols]
    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
    
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect weather data from NDBC buoys.")
    parser.add_argument(
        "--station",
        default="46042",
        help="NDBC station ID (default: 46042 Monterey Bay)"
    )
    parser.add_argument(
        "--start",
        help="Start date YYYY-MM-DD (default: 1 year ago)"
    )
    parser.add_argument(
        "--end",
        help="End date YYYY-MM-DD (default: today)"
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
        default=1.0,
        help="Seconds between API calls (default: 1.0)"
    )
    
    args = parser.parse_args(argv)
    
    print(f"Collecting weather data for NDBC station {args.station}")
    
    # If no date range specified, get last 45 days (realtime) + last year (historical)
    if not args.start:
        print("\nFetching realtime data (last 45 days)...")
        try:
            realtime_df = fetch_ndbc_realtime(args.station)
            realtime_df = process_ndbc_data(realtime_df)
            print(f"  Collected {len(realtime_df)} realtime records")
        except Exception as e:
            print(f"  Error fetching realtime: {e}")
            realtime_df = pd.DataFrame()
        
        # Also fetch historical data for past year
        print("\nFetching historical data (past 12 months)...")
        historical_dfs = []
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        current = start_date
        while current <= end_date:
            year = current.year
            month = current.month
            
            print(f"  Fetching {year}-{month:02d}...", end=" ")
            try:
                df = fetch_ndbc_historical_month(args.station, year, month)
                if not df.empty:
                    historical_dfs.append(df)
                    print(f"{len(df)} records")
                else:
                    print("no data")
            except Exception as e:
                print(f"error: {e}")
            
            # Move to next month
            if month == 12:
                current = datetime(year + 1, 1, 1)
            else:
                current = datetime(year, month + 1, 1)
            
            time.sleep(args.pause)
        
        # Combine all data
        all_dfs = []
        if not realtime_df.empty:
            all_dfs.append(realtime_df)
        if historical_dfs:
            all_dfs.extend(historical_dfs)
        
        if not all_dfs:
            print("\nNo data collected!")
            return 1
        
        combined = pd.concat(all_dfs, ignore_index=True)
        
    else:
        # Custom date range
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now()
        
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        
        historical_dfs = []
        current = start_date
        
        while current <= end_date:
            year = current.year
            month = current.month
            
            print(f"Fetching {year}-{month:02d}...", end=" ")
            try:
                df = fetch_ndbc_historical_month(args.station, year, month)
                if not df.empty:
                    historical_dfs.append(df)
                    print(f"{len(df)} records")
                else:
                    print("no data")
            except Exception as e:
                print(f"error: {e}")
            
            if month == 12:
                current = datetime(year + 1, 1, 1)
            else:
                current = datetime(year, month + 1, 1)
            
            time.sleep(args.pause)
        
        if not historical_dfs:
            print("No data collected!")
            return 1
        
        combined = pd.concat(historical_dfs, ignore_index=True)
    
    # Process and deduplicate
    combined = process_ndbc_data(combined)
    combined = combined.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    
    print(f"\nTotal unique records: {len(combined)}")
    print(f"Date range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")
    
    # Save to CSV
    args.outdir.mkdir(parents=True, exist_ok=True)
    output_file = args.outdir / f"{args.station}_weather_data.csv"
    combined.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")
    
    # Print summary
    print("\nWeather Summary:")
    if 'wind_speed_kt' in combined.columns:
        print(f"  Wind speed: {combined['wind_speed_kt'].min():.1f} - {combined['wind_speed_kt'].max():.1f} kt")
    if 'air_temp_f' in combined.columns:
        print(f"  Air temp: {combined['air_temp_f'].min():.1f} - {combined['air_temp_f'].max():.1f}°F")
    if 'water_temp_f' in combined.columns:
        print(f"  Water temp: {combined['water_temp_f'].min():.1f} - {combined['water_temp_f'].max():.1f}°F")
    if 'upwelling_favorable' in combined.columns:
        upwelling_pct = combined['upwelling_favorable'].mean() * 100
        print(f"  Upwelling-favorable winds: {upwelling_pct:.1f}% of observations")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
