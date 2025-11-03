#!/usr/bin/env python3
"""
Feature engineering for fishing prediction model.

Creates ML-ready features from raw environmental data:
- Temperature gradients and rates of change
- Pressure trends and stability metrics
- Temporal features (hour, day of week, season)
- Tidal cycle indicators
- Historical rolling statistics
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np


def load_temperature_data(path: Path) -> pd.DataFrame:
    """Load water temperature CSV."""
    df = pd.read_csv(path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def load_pressure_data(path: Path) -> pd.DataFrame:
    """Load barometric pressure CSV."""
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def load_weather_data(path: Path) -> pd.DataFrame:
    """Load NDBC weather data CSV."""
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def load_tidal_data(path: Path) -> pd.DataFrame:
    """Load tidal data CSV."""
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from timestamp."""
    df = df.copy()
    
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['month'] = df['timestamp'].dt.month
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week
    
    # Cyclical encoding for periodic features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Season (meteorological)
    df['season'] = ((df['month'] % 12) // 3).map({
        0: 'winter',
        1: 'spring', 
        2: 'summer',
        3: 'fall'
    })
    
    # Early morning flag (prime fishing time)
    df['is_early_morning'] = ((df['hour'] >= 5) & (df['hour'] <= 9)).astype(int)
    
    # Weekend flag
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    return df


def create_temperature_features(df: pd.DataFrame, temp_col: str = None) -> pd.DataFrame:
    """Create features from temperature data."""
    df = df.copy()
    
    # Auto-detect temperature column
    if temp_col is None:
        if 'morning_temp_avg_F' in df.columns:
            temp_col = 'morning_temp_avg_F'
        elif 'morning_temp_F' in df.columns:
            temp_col = 'morning_temp_F'
        else:
            return df
    
    if temp_col not in df.columns:
        return df
    
    # Temperature change over different windows
    for window in [1, 3, 7, 14]:
        df[f'temp_change_{window}d'] = df[temp_col].diff(window)
        df[f'temp_pct_change_{window}d'] = df[temp_col].pct_change(window) * 100
    
    # Rolling statistics
    for window in [3, 7, 14, 30]:
        df[f'temp_rolling_mean_{window}d'] = df[temp_col].rolling(window, min_periods=1).mean()
        df[f'temp_rolling_std_{window}d'] = df[temp_col].rolling(window, min_periods=1).std()
        df[f'temp_rolling_min_{window}d'] = df[temp_col].rolling(window, min_periods=1).min()
        df[f'temp_rolling_max_{window}d'] = df[temp_col].rolling(window, min_periods=1).max()
    
    # Temperature volatility (CV)
    df['temp_volatility_7d'] = (df['temp_rolling_std_7d'] / df['temp_rolling_mean_7d']) * 100
    
    # Anomaly from rolling average
    df['temp_anomaly_7d'] = df[temp_col] - df['temp_rolling_mean_7d']
    df['temp_anomaly_30d'] = df[temp_col] - df['temp_rolling_mean_30d']
    
    # Optimal temperature ranges (species-specific, example for rockfish)
    df['temp_in_optimal_range'] = ((df[temp_col] >= 50) & (df[temp_col] <= 58)).astype(int)
    
    return df


def create_pressure_features(df: pd.DataFrame, pressure_col: str = 'pressure_mb') -> pd.DataFrame:
    """Create features from barometric pressure."""
    df = df.copy()
    
    if pressure_col not in df.columns:
        return df
    
    # Pressure trends
    for window in [1, 3, 6, 12, 24]:
        df[f'pressure_change_{window}h'] = df[pressure_col].diff(window)
    
    # Pressure stability (std over window)
    for window in [6, 12, 24]:
        df[f'pressure_stability_{window}h'] = df[pressure_col].rolling(window, min_periods=1).std()
    
    # Classify pressure trend
    df['pressure_trend_6h'] = pd.cut(
        df['pressure_change_6h'],
        bins=[-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf],
        labels=['rapid_fall', 'falling', 'stable', 'rising', 'rapid_rise']
    )
    
    # High/low pressure flags (relative)
    pressure_median = df[pressure_col].median()
    df['is_high_pressure'] = (df[pressure_col] > pressure_median + 2).astype(int)
    df['is_low_pressure'] = (df[pressure_col] < pressure_median - 2).astype(int)
    
    return df


def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features from weather data."""
    df = df.copy()
    
    # Wind features
    if 'wind_speed_kt' in df.columns:
        for window in [6, 12, 24]:
            df[f'wind_speed_mean_{window}h'] = df['wind_speed_kt'].rolling(window, min_periods=1).mean()
            df[f'wind_speed_max_{window}h'] = df['wind_speed_kt'].rolling(window, min_periods=1).max()
    
    # Upwelling features (critical for SST prediction)
    if 'wind_north_component_kt' in df.columns:
        # Cumulative upwelling index (running sum of upwelling-favorable winds)
        df['upwelling_index_24h'] = df['wind_north_component_kt'].rolling(24, min_periods=1).sum()
        df['upwelling_index_72h'] = df['wind_north_component_kt'].rolling(72, min_periods=1).sum()
        
        # Upwelling persistence (hours of continuous upwelling winds)
        df['upwelling_hours'] = (df['wind_north_component_kt'] < -5).astype(int)
        df['upwelling_hours_24h'] = df['upwelling_hours'].rolling(24, min_periods=1).sum()
    
    # Air-sea temperature difference (drives heat exchange)
    if 'air_temp_f' in df.columns and 'water_temp_f' in df.columns:
        df['air_sea_temp_diff'] = df['air_temp_f'] - df['water_temp_f']
        df['air_sea_temp_diff_mean_24h'] = df['air_sea_temp_diff'].rolling(24, min_periods=1).mean()
    
    # Wave energy (mixing indicator)
    if 'wave_height_m' in df.columns:
        df['wave_energy'] = df['wave_height_m'] ** 2  # Proportional to wave energy
        df['wave_energy_mean_24h'] = df['wave_energy'].rolling(24, min_periods=1).mean()
    
    return df


def create_tidal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features from tidal data."""
    df = df.copy()
    
    if 'tidal_phase' not in df.columns:
        return df
    
    # One-hot encode tidal phase
    phase_dummies = pd.get_dummies(df['tidal_phase'], prefix='tide')
    df = pd.concat([df, phase_dummies], axis=1)
    
    # Tidal range (high-low amplitude)
    if 'tidal_range_ft' in df.columns:
        df['is_spring_tide'] = (df['tidal_range_ft'] > df['tidal_range_ft'].quantile(0.75)).astype(int)
        df['is_neap_tide'] = (df['tidal_range_ft'] < df['tidal_range_ft'].quantile(0.25)).astype(int)
    
    # Tidal current strength (rate of change)
    if 'tidal_rate_ft_per_hr' in df.columns:
        df['tidal_current_strong'] = (abs(df['tidal_rate_ft_per_hr']) > 0.3).astype(int)
    
    # Prime fishing time: incoming tide + early morning
    if 'is_early_morning' in df.columns and 'tide_flood' in df.columns:
        df['prime_tide_time'] = (df['is_early_morning'] * df['tide_flood']).astype(int)
    
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction terms between features."""
    df = df.copy()
    
    # Temperature-pressure interaction
    if 'morning_temp_avg_F' in df.columns and 'pressure_mb' in df.columns:
        df['temp_pressure_product'] = df['morning_temp_avg_F'] * df['pressure_mb']
    
    # Temperature change during pressure change
    if 'temp_change_1d' in df.columns and 'pressure_change_24h' in df.columns:
        df['temp_pressure_change_interaction'] = df['temp_change_1d'] * df['pressure_change_24h']
    
    # Early morning + stable pressure (ideal conditions)
    if 'is_early_morning' in df.columns and 'pressure_stability_6h' in df.columns:
        df['optimal_time_stable_pressure'] = (
            df['is_early_morning'] * (df['pressure_stability_6h'] < df['pressure_stability_6h'].median())
        ).astype(int)
    
    return df


def create_lag_features(df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
    """Create lagged versions of specified columns."""
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    return df


def load_weather_data(path: Path) -> pd.DataFrame:
    """Load NDBC weather data CSV."""
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def load_tidal_data(path: Path) -> pd.DataFrame:
    """Load tidal data CSV."""
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features from weather data."""
    df = df.copy()
    
    # Wind features
    if 'wind_speed_kt' in df.columns:
        for window in [6, 12, 24]:
            df[f'wind_speed_mean_{window}h'] = df['wind_speed_kt'].rolling(window, min_periods=1).mean()
            df[f'wind_speed_max_{window}h'] = df['wind_speed_kt'].rolling(window, min_periods=1).max()
    
    # Upwelling features (critical for SST prediction)
    if 'wind_north_component_kt' in df.columns:
        # Cumulative upwelling index (running sum of upwelling-favorable winds)
        df['upwelling_index_24h'] = df['wind_north_component_kt'].rolling(24, min_periods=1).sum()
        df['upwelling_index_72h'] = df['wind_north_component_kt'].rolling(72, min_periods=1).sum()
        
        # Upwelling persistence (hours of continuous upwelling winds)
        df['upwelling_hours'] = (df['wind_north_component_kt'] < -5).astype(int)
        df['upwelling_hours_24h'] = df['upwelling_hours'].rolling(24, min_periods=1).sum()
    
    # Air-sea temperature difference (drives heat exchange)
    if 'air_temp_f' in df.columns and 'water_temp_f' in df.columns:
        df['air_sea_temp_diff'] = df['air_temp_f'] - df['water_temp_f']
        df['air_sea_temp_diff_mean_24h'] = df['air_sea_temp_diff'].rolling(24, min_periods=1).mean()
    
    # Wave energy (mixing indicator)
    if 'wave_height_m' in df.columns:
        df['wave_energy'] = df['wave_height_m'] ** 2  # Proportional to wave energy
        df['wave_energy_mean_24h'] = df['wave_energy'].rolling(24, min_periods=1).mean()
    
    return df


def create_tidal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features from tidal data."""
    df = df.copy()
    
    if 'tidal_phase' not in df.columns:
        return df
    
    # One-hot encode tidal phase
    phase_dummies = pd.get_dummies(df['tidal_phase'], prefix='tide')
    df = pd.concat([df, phase_dummies], axis=1)
    
    # Tidal range (high-low amplitude)
    if 'tidal_range_ft' in df.columns:
        df['is_spring_tide'] = (df['tidal_range_ft'] > df['tidal_range_ft'].quantile(0.75)).astype(int)
        df['is_neap_tide'] = (df['tidal_range_ft'] < df['tidal_range_ft'].quantile(0.25)).astype(int)
    
    # Tidal current strength (rate of change)
    if 'tidal_rate_ft_per_hr' in df.columns:
        df['tidal_current_strong'] = (abs(df['tidal_rate_ft_per_hr']) > 0.3).astype(int)
    
    # Prime fishing time: incoming tide + early morning
    if 'is_early_morning' in df.columns and 'tide_flood' in df.columns:
        df['prime_tide_time'] = (df['is_early_morning'] * df['tide_flood']).astype(int)
    
    return df


def merge_datasets(temp_df: pd.DataFrame, pressure_df: pd.DataFrame, 
                   weather_df: pd.DataFrame | None = None,
                   tidal_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Merge temperature, pressure, weather, and tidal datasets on timestamp."""
    
    # Ensure temperature df has timestamp
    if 'timestamp' not in temp_df.columns:
        if 'week_start' in temp_df.columns:
            temp_df['timestamp'] = pd.to_datetime(temp_df['week_start'])
        elif 'date' in temp_df.columns:
            temp_df['timestamp'] = pd.to_datetime(temp_df['date'])
        else:
            raise ValueError("Temperature data must have 'timestamp', 'week_start', or 'date' column")
    
    if 'timestamp' not in pressure_df.columns:
        pressure_df['timestamp'] = pd.to_datetime(pressure_df['timestamp'])
    
    # Start with temp-pressure merge
    merged = pd.merge_asof(
        temp_df.sort_values('timestamp'),
        pressure_df.sort_values('timestamp'),
        on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta('12 hours')
    )
    
    # Merge weather data (hourly observations)
    if weather_df is not None:
        if 'timestamp' not in weather_df.columns:
            weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
        
        # Aggregate weather to daily (morning snapshot)
        weather_daily = weather_df.set_index('timestamp').resample('D').agg({
            col: 'mean' for col in weather_df.columns 
            if col != 'timestamp' and pd.api.types.is_numeric_dtype(weather_df[col])
        }).reset_index()
        
        merged = pd.merge_asof(
            merged.sort_values('timestamp'),
            weather_daily.sort_values('timestamp'),
            on='timestamp',
            direction='nearest',
            tolerance=pd.Timedelta('24 hours'),
            suffixes=('', '_weather')
        )
    
    # Merge tidal data
    if tidal_df is not None:
        if 'timestamp' not in tidal_df.columns:
            tidal_df['timestamp'] = pd.to_datetime(tidal_df['timestamp'])
        
        # Aggregate tidal to daily (morning snapshot around 6am)
        tidal_df['hour'] = tidal_df['timestamp'].dt.hour
        morning_tidal = tidal_df[tidal_df['hour'].between(5, 9)].copy()
        
        if len(morning_tidal) > 0:
            tidal_daily = morning_tidal.set_index('timestamp').resample('D').first().reset_index()
            
            merged = pd.merge_asof(
                merged.sort_values('timestamp'),
                tidal_daily.sort_values('timestamp'),
                on='timestamp',
                direction='nearest',
                tolerance=pd.Timedelta('24 hours'),
                suffixes=('', '_tidal')
            )
    
    return merged


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Engineer features for fishing prediction.")
    parser.add_argument(
        "--temp-file",
        type=Path,
        default=Path("data/processed/9413450_morning_daily.csv"),
        help="Temperature data CSV"
    )
    parser.add_argument(
        "--pressure-file",
        type=Path,
        default=Path("data/processed/9413450_barometric_pressure.csv"),
        help="Pressure data CSV"
    )
    parser.add_argument(
        "--weather-file",
        type=Path,
        default=Path("data/processed/46042_weather_data.csv"),
        help="NDBC weather data CSV"
    )
    parser.add_argument(
        "--tidal-file",
        type=Path,
        default=Path("data/processed/9413450_tidal_data.csv"),
        help="Tidal data CSV"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/features/fishing_features.csv"),
        help="Output feature CSV"
    )
    
    args = parser.parse_args(argv)
    
    print("Loading data...")
    
    # Load datasets
    if not args.temp_file.exists():
        print(f"Error: Temperature file not found: {args.temp_file}")
        return 1
    
    temp_df = load_temperature_data(args.temp_file)
    print(f"Loaded {len(temp_df)} temperature records")
    
    # Load pressure (optional)
    pressure_df = None
    if args.pressure_file.exists():
        pressure_df = load_pressure_data(args.pressure_file)
        print(f"Loaded {len(pressure_df)} pressure records")
    else:
        print("Warning: Pressure file not found")
        pressure_df = pd.DataFrame({'timestamp': temp_df['timestamp'] if 'timestamp' in temp_df.columns else pd.to_datetime(temp_df['week_start'])})
    
    # Load weather (optional)
    weather_df = None
    if args.weather_file.exists():
        weather_df = load_weather_data(args.weather_file)
        print(f"Loaded {len(weather_df)} weather records")
    else:
        print("Warning: Weather file not found (run collect_weather_data.py)")
    
    # Load tidal (optional)
    tidal_df = None
    if args.tidal_file.exists():
        tidal_df = load_tidal_data(args.tidal_file)
        print(f"Loaded {len(tidal_df)} tidal records")
    else:
        print("Warning: Tidal file not found")
    
    # Merge all datasets
    print("Merging datasets...")
    df = merge_datasets(temp_df, pressure_df, weather_df, tidal_df)
    print(f"Combined dataset: {len(df)} records")
    
    # Feature engineering pipeline
    print("\nCreating features...")
    
    df = create_temporal_features(df)
    print("  ✓ Temporal features")
    
    df = create_temperature_features(df)
    print("  ✓ Temperature features")
    
    if 'pressure_mb' in df.columns:
        df = create_pressure_features(df)
        print("  ✓ Pressure features")
    
    if weather_df is not None:
        df = create_weather_features(df)
        print("  ✓ Weather features (wind, upwelling)")
    
    if tidal_df is not None:
        df = create_tidal_features(df)
        print("  ✓ Tidal features")
    
    df = create_interaction_features(df)
    print("  ✓ Interaction features")
    
    # Optional: Create lag features for time-series models
    lag_cols = ['morning_temp_avg_F']
    if 'pressure_mb' in df.columns:
        lag_cols.append('pressure_mb')
    
    df = create_lag_features(df, lag_cols, lags=[1, 7, 14])
    print("  ✓ Lag features")
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    
    print(f"\n✅ Features saved to {args.output}")
    print(f"Total features: {len(df.columns)}")
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Show feature summary
    print("\nFeature categories:")
    temporal = [c for c in df.columns if any(x in c for x in ['hour', 'day', 'week', 'month', 'season'])]
    temp = [c for c in df.columns if 'temp' in c]
    pressure = [c for c in df.columns if 'pressure' in c]
    wind = [c for c in df.columns if 'wind' in c or 'upwelling' in c]
    tidal = [c for c in df.columns if 'tide' in c or 'tidal' in c]
    
    print(f"  Temporal: {len(temporal)}")
    print(f"  Temperature: {len(temp)}")
    print(f"  Pressure: {len(pressure)}")
    print(f"  Wind/Upwelling: {len(wind)}")
    print(f"  Tidal: {len(tidal)}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
