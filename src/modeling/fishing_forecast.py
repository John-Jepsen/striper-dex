#!/usr/bin/env python3
"""
Fishing forecast for Monterey Bay based on water temperature patterns.

This script analyzes historical SST data to predict favorable fishing conditions
for the upcoming week by examining:
- Current temperature vs. historical patterns
- Recent temperature trends
- Optimal temperature ranges for target species
- Day-of-week historical catch correlations
"""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor  # type: ignore[import]
from sklearn.preprocessing import StandardScaler  # type: ignore[import]
from sklearn.model_selection import TimeSeriesSplit  # type: ignore[import]
from sklearn.metrics import mean_absolute_error, mean_squared_error  # type: ignore[import]


@dataclass
class SpeciesProfile:
    """Temperature and seasonal preferences for target species."""

    name: str
    optimal_temp_range: tuple[float, float]  # Fahrenheit
    preferred_months: list[int]  # 1-12
    activity_level: Literal["high", "medium", "low"]


MONTEREY_SPECIES = [
    SpeciesProfile("Rockfish", (52, 58), [5, 6, 7, 8, 9, 10], "high"),
    SpeciesProfile("Lingcod", (48, 56), [4, 5, 6, 7, 8, 9], "high"),
    SpeciesProfile("Salmon (King)", (52, 58), [4, 5, 6, 7, 8, 9, 10], "high"),
    SpeciesProfile("Halibut", (55, 62), [5, 6, 7, 8, 9, 10], "medium"),
    SpeciesProfile("Striped Bass", (58, 65), [5, 6, 7, 8, 9], "medium"),
    SpeciesProfile("Surfperch", (50, 60), list(range(1, 13)), "high"),
]


def load_daily_data(path: Path) -> pd.DataFrame:
    """Load daily morning temperature snapshots."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal and statistical features for forecasting."""
    df = df.copy()
    
    # Temporal features (avoid raw month to prevent overfitting to calendar)
    df["year"] = df["date"].dt.year
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["date"].dt.dayofweek  # Monday=0, Sunday=6
    
    # Cyclical encoding for seasonality (use day-of-year instead of month)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    
    # Lagged temperature features (critical for temperature continuity)
    for lag in [1, 2, 3, 5, 7, 10, 14, 21, 30]:
        df[f"temp_lag_{lag}"] = df["morning_temp_F"].shift(lag)
    
    # Rolling statistics (emphasize recent temperature behavior)
    for window in [3, 5, 7, 10, 14, 21, 30, 60, 90]:
        df[f"temp_roll_mean_{window}"] = (
            df["morning_temp_F"].rolling(window, min_periods=1).mean()
        )
        df[f"temp_roll_std_{window}"] = (
            df["morning_temp_F"].rolling(window, min_periods=1).std()
        )
        df[f"temp_roll_min_{window}"] = (
            df["morning_temp_F"].rolling(window, min_periods=1).min()
        )
        df[f"temp_roll_max_{window}"] = (
            df["morning_temp_F"].rolling(window, min_periods=1).max()
        )
    
    # Temperature change rates (velocity and acceleration)
    df["temp_change_1d"] = df["morning_temp_F"].diff(1)
    df["temp_change_3d"] = df["morning_temp_F"].diff(3)
    df["temp_change_7d"] = df["morning_temp_F"].diff(7)
    df["temp_change_14d"] = df["morning_temp_F"].diff(14)
    df["temp_change_30d"] = df["morning_temp_F"].diff(30)
    
    # Temperature acceleration (second derivative)
    df["temp_accel_1d"] = df["temp_change_1d"].diff(1)
    df["temp_accel_7d"] = df["temp_change_7d"].diff(7)
    
    # Historical climatology for this day of year
    df["climatology"] = df.groupby("day_of_year")["morning_temp_F"].transform("median")
    df["anomaly"] = df["morning_temp_F"] - df["climatology"]
    
    return df


def train_temperature_model(df: pd.DataFrame) -> tuple[RandomForestRegressor, StandardScaler, list[str], dict]:
    """Train a model to forecast temperature N days ahead with cross-validation."""
    
    # Only use complete cases for training
    df_complete = df.dropna().copy()
    
    # Features for prediction
    feature_cols = [
        col for col in df_complete.columns
        if col not in ["date", "morning_temp_F", "year"]
    ]
    
    X = df_complete[feature_cols].values
    y = df_complete["morning_temp_F"].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Time-series cross-validation
    print("\nPerforming time-series cross-validation...")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled), 1):
        X_train_cv, X_val_cv = X_scaled[train_idx], X_scaled[val_idx]
        y_train_cv, y_val_cv = y[train_idx], y[val_idx]
        
        model_cv = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features=0.4,
            random_state=42,
            n_jobs=-1,
        )
        model_cv.fit(X_train_cv, y_train_cv)
        
        y_pred_cv = model_cv.predict(X_val_cv)
        mae = mean_absolute_error(y_val_cv, y_pred_cv)
        rmse = np.sqrt(mean_squared_error(y_val_cv, y_pred_cv))
        
        cv_scores.append({'fold': fold, 'mae': mae, 'rmse': rmse})
        print(f"  Fold {fold}: MAE={mae:.2f}°F, RMSE={rmse:.2f}°F")
    
    avg_mae = np.mean([s['mae'] for s in cv_scores])
    avg_rmse = np.mean([s['rmse'] for s in cv_scores])
    print(f"\n  Average: MAE={avg_mae:.2f}°F, RMSE={avg_rmse:.2f}°F")
    
    # Train final model on all data
    print("\nTraining final model on all data...")
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features=0.4,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled, y)
    
    validation_metrics = {
        'cv_mae_mean': avg_mae,
        'cv_rmse_mean': avg_rmse,
        'cv_scores': cv_scores,
    }
    
    return model, scaler, feature_cols, validation_metrics


def forecast_next_week(
    df: pd.DataFrame,
    model: RandomForestRegressor,
    scaler: StandardScaler,
    feature_cols: list[str],
    forecast_days: int = 7,
) -> pd.DataFrame:
    """Generate temperature forecasts for the next N days with uncertainty estimates."""
    
    last_date = df["date"].max()
    forecast_dates = [last_date + dt.timedelta(days=i) for i in range(1, forecast_days + 1)]
    
    forecasts = []
    df_extended = df.copy()
    
    for forecast_date in forecast_dates:
        # Create row for forecast date with temporal features
        new_row = pd.DataFrame({
            "date": [forecast_date],
            "morning_temp_F": [np.nan],
        })
        
        # Append and re-engineer features
        df_temp = pd.concat([df_extended, new_row], ignore_index=True)
        df_temp = engineer_features(df_temp)
        
        # Get features for the new row
        new_row_idx = len(df_temp) - 1
        X_new_series = df_temp.loc[new_row_idx, feature_cols]
        X_new = np.array(X_new_series.values, dtype=float).reshape(1, -1)
        
        # Handle any remaining NaN values by forward-filling from most recent
        if np.any(np.isnan(X_new)):
            last_complete_series = df_temp[feature_cols].ffill().iloc[-1]
            last_complete = np.array(last_complete_series.values, dtype=float).reshape(1, -1)
            nan_mask = np.isnan(X_new)
            X_new[nan_mask] = last_complete[nan_mask]
        
        # Predict with uncertainty (per-tree predictions)
        X_scaled = scaler.transform(X_new)
        
        # Get predictions from all trees for uncertainty estimation
        tree_predictions = np.array([tree.predict(X_scaled)[0] for tree in model.estimators_])
        temp_pred = tree_predictions.mean()
        temp_std = tree_predictions.std()
        temp_lower = np.percentile(tree_predictions, 5)   # 90% CI lower
        temp_upper = np.percentile(tree_predictions, 95)  # 90% CI upper
        
        # Update the extended dataframe with prediction
        df_temp.loc[new_row_idx, "morning_temp_F"] = temp_pred
        df_extended = df_temp.copy()
        
        forecasts.append({
            "date": forecast_date,
            "predicted_temp_F": temp_pred,
            "uncertainty_std": temp_std,
            "ci_lower_90": temp_lower,
            "ci_upper_90": temp_upper,
        })
    
    return pd.DataFrame(forecasts)


def score_fishing_conditions(
    forecast_df: pd.DataFrame,
    species_list: Sequence[SpeciesProfile],
) -> pd.DataFrame:
    """Score each forecasted day for fishing quality by species."""
    
    scored = forecast_df.copy()
    scored["day_of_week"] = pd.to_datetime(scored["date"]).dt.dayofweek
    scored["day_name"] = pd.to_datetime(scored["date"]).dt.day_name()
    scored["month"] = pd.to_datetime(scored["date"]).dt.month
    scored["hour"] = pd.to_datetime(scored["date"]).dt.hour
    
    for species in species_list:
        # Temperature score (0-100)
        temp = scored["predicted_temp_F"]
        opt_low, opt_high = species.optimal_temp_range
        
        # Gaussian-like scoring centered on optimal range
        range_center = (opt_low + opt_high) / 2
        range_width = opt_high - opt_low
        
        temp_score = 100 * np.exp(-((temp - range_center) ** 2) / (2 * (range_width / 2) ** 2))
        temp_score = np.clip(temp_score, 0, 100)
        
        # Seasonal score (0-100)
        seasonal_score = scored["month"].apply(
            lambda m: 100 if m in species.preferred_months else 30
        )
        
        # Tidal bonus (if available)
        tidal_bonus = 0
        if 'tidal_phase' in scored.columns:
            # Flood/ebb tide = active feeding (+15 points)
            tidal_bonus = scored['tidal_phase'].apply(
                lambda phase: 15 if phase in ['flood', 'ebb'] else 0
            )
        elif 'tide_flood' in scored.columns or 'tide_ebb' in scored.columns:
            flood = scored.get('tide_flood', 0)
            ebb = scored.get('tide_ebb', 0)
            tidal_bonus = (flood + ebb) * 15
        
        # Time of day bonus (early morning +10 points)
        time_bonus = 0
        if 'hour' in scored.columns:
            time_bonus = scored['hour'].apply(lambda h: 10 if 5 <= h <= 9 else 0)
        
        # Combined score with bonuses
        combined_score = 0.6 * temp_score + 0.25 * seasonal_score + tidal_bonus + time_bonus
        combined_score = np.clip(combined_score, 0, 100)
        
        scored[f"{species.name.lower().replace(' ', '_')}_score"] = combined_score.round(1)
    
    return scored


def print_forecast_report(
    forecast_df: pd.DataFrame,
    current_temp: float,
    species_list: Sequence[SpeciesProfile],
) -> None:
    """Print human-readable fishing forecast."""
    
    print("\n" + "=" * 80)
    print("🎣 MONTEREY BAY FISHING FORECAST")
    print("=" * 80)
    print(f"\nCurrent water temperature: {current_temp:.1f}°F")
    print(f"Forecast generated: {dt.datetime.now():%Y-%m-%d %H:%M}")
    print("\n" + "-" * 80)
    
    for _, row in forecast_df.iterrows():
        date = pd.to_datetime(row["date"])
        temp = row["predicted_temp_F"]
        ci_lower = row.get("ci_lower_90", temp)
        ci_upper = row.get("ci_upper_90", temp)
        uncertainty = row.get("uncertainty_std", 0)
        day_name = row["day_name"]
        
        # Confidence indicator
        if uncertainty < 0.5:
            confidence = "🟢 High"
        elif uncertainty < 1.0:
            confidence = "🟡 Medium"
        else:
            confidence = "🔴 Low"
        
        print(f"\n📅 {date:%A, %B %d, %Y} ({day_name})")
        print(f"   Predicted temp: {temp:.1f}°F (90% CI: {ci_lower:.1f}-{ci_upper:.1f}°F)")
        print(f"   Forecast confidence: {confidence} (±{uncertainty:.1f}°F)")
        
        # Find best species for this day
        species_scores = []
        for species in species_list:
            col_name = f"{species.name.lower().replace(' ', '_')}_score"
            if col_name in row:
                species_scores.append((species.name, row[col_name]))
        
        species_scores.sort(key=lambda x: x[1], reverse=True)
        
        print("   Target species (ranked by conditions):")
        for species_name, score in species_scores[:3]:
            if score >= 70:
                emoji = "🟢"
                rating = "Excellent"
            elif score >= 50:
                emoji = "🟡"
                rating = "Good"
            else:
                emoji = "🔴"
                rating = "Fair"
            print(f"      {emoji} {species_name}: {score:.0f}/100 ({rating})")
    
    # Overall recommendation
    print("\n" + "-" * 80)
    print("📊 WEEKLY SUMMARY")
    print("-" * 80)
    
    best_day = forecast_df.loc[
        forecast_df[[col for col in forecast_df.columns if col.endswith("_score")]].mean(axis=1).idxmax()
    ]
    best_date = pd.to_datetime(best_day["date"])
    best_temp = best_day["predicted_temp_F"]
    
    print(f"\n🌟 Best overall day: {best_date:%A, %B %d}")
    print(f"   Expected temp: {best_temp:.1f}°F")
    
    temp_trend = forecast_df["predicted_temp_F"].iloc[-1] - forecast_df["predicted_temp_F"].iloc[0]
    trend_emoji = "⬆️" if temp_trend > 0.5 else "⬇️" if temp_trend < -0.5 else "➡️"
    print(f"   Week trend: {trend_emoji} {temp_trend:+.1f}°F change expected")
    print("\n" + "=" * 80 + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate fishing forecast based on SST predictions."
    )
    parser.add_argument(
        "--daily-data",
        type=Path,
        default=Path("data/processed/9413450_morning_daily.csv"),
        help="Daily temperature CSV from pull_noaa_history.py",
    )
    parser.add_argument(
        "--forecast-days",
        type=int,
        default=7,
        help="Number of days to forecast (default: 7)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV path to save forecast results",
    )
    
    args = parser.parse_args(argv)
    
    # Load and prepare data
    print("Loading daily temperature data...")
    df = load_daily_data(args.daily_data)
    current_temp = df["morning_temp_F"].iloc[-1]
    
    print("Engineering features...")
    df_features = engineer_features(df)
    
    print("Training temperature forecasting model...")
    model, scaler, feature_cols, validation_metrics = train_temperature_model(df_features)
    
    print(f"Generating {args.forecast_days}-day forecast...")
    forecast = forecast_next_week(df_features, model, scaler, feature_cols, args.forecast_days)
    
    print("Scoring fishing conditions by species...")
    forecast_scored = score_fishing_conditions(forecast, MONTEREY_SPECIES)
    
    # Print report
    print_forecast_report(forecast_scored, current_temp, MONTEREY_SPECIES, validation_metrics)
    
    # Save if requested
    if args.output:
        forecast_scored.to_csv(args.output, index=False)
        print(f"Forecast saved to: {args.output}")


if __name__ == "__main__":
    main()
