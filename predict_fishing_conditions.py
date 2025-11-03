#!/usr/bin/env python3
"""
Generate predictions for optimal fishing times using trained model.

Usage:
    # Today's conditions
    python predict_fishing_conditions.py --date today
    
    # Specific date
    python predict_fishing_conditions.py --date 2024-11-15
    
    # 7-day forecast
    python predict_fishing_conditions.py --forecast 7
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict
import warnings

import pandas as pd
import numpy as np
import joblib

warnings.filterwarnings('ignore')


def load_model(model_dir: Path) -> tuple:
    """Load trained model, scaler, and metadata."""
    model = joblib.load(model_dir / 'fishing_model.joblib')
    scaler = joblib.load(model_dir / 'scaler.joblib')
    
    import json
    with open(model_dir / 'model_metadata.json') as f:
        metadata = json.load(f)
    
    return model, scaler, metadata


def fetch_current_conditions(station: str = "9413450", pred_date: dt.datetime = None) -> Dict:
    """
    Fetch environmental conditions from historical data patterns.
    Uses historical averages for the same day-of-year.
    """
    if pred_date is None:
        pred_date = dt.datetime.now()
    
    # Load historical features to use real patterns
    features_path = Path('data/features/fishing_features.csv')
    if not features_path.exists():
        print(f"Warning: No historical data found, using defaults")
        return {
            'timestamp': pred_date,
            'morning_temp_F': 55.0,
            'pressure_mb': 1015.0,
            'pressure_change_6h': 0.5,
            'hour': 7,
            'day_of_week': pred_date.weekday(),
            'month': pred_date.month,
        }
    
    df = pd.read_csv(features_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Find historical data for same day of year (±7 days window)
    target_doy = pred_date.timetuple().tm_yday
    same_doy = df[df['date'].dt.dayofyear.between(target_doy - 7, target_doy + 7)]
    
    if len(same_doy) == 0:
        # Fall back to same month
        same_doy = df[df['date'].dt.month == pred_date.month]
    
    if len(same_doy) > 0:
        # Use median values from historical patterns (numeric columns only)
        numeric_cols = same_doy.select_dtypes(include=[np.number]).columns
        conditions = same_doy[numeric_cols].median().to_dict()
        conditions['timestamp'] = pred_date
        conditions['hour'] = 7
        conditions['day_of_week'] = pred_date.weekday()
        conditions['month'] = pred_date.month
        return conditions
    
    return {
        'timestamp': pred_date,
        'morning_temp_F': 55.0,
        'pressure_mb': 1015.0,
        'pressure_change_6h': 0.5,
        'hour': 7,
        'day_of_week': pred_date.weekday(),
        'month': pred_date.month,
    }


def prepare_prediction_features(conditions: Dict, feature_names: list) -> pd.DataFrame:
    """Convert conditions dict to feature DataFrame matching training schema."""
    
    # Start with provided conditions
    features = conditions.copy()
    
    # Calculate derived features if needed
    hour = features.get('hour', 7)
    month = features.get('month', 1)
    
    features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    features['month_sin'] = np.sin(2 * np.pi * month / 12)
    features['month_cos'] = np.cos(2 * np.pi * month / 12)
    
    features['is_early_morning'] = 1 if 5 <= hour <= 9 else 0
    features['is_weekend'] = 1 if features.get('day_of_week', 0) >= 5 else 0
    
    # Pressure thresholds
    pressure = features.get('pressure_mb', 1015.0)
    features['is_high_pressure'] = 1 if pressure > 1020 else 0
    features['is_low_pressure'] = 1 if pressure < 1010 else 0
    
    # Temperature optimal range (60-70F for striped bass)
    temp = features.get('morning_temp_F', 55.0)
    features['temp_in_optimal_range'] = 1 if 60 <= temp <= 70 else 0
    
    # Create DataFrame with all required features
    df = pd.DataFrame([features])
    
    # Fill missing features with defaults
    for feat in feature_names:
        if feat not in df.columns:
            df[feat] = 0
    
    # Select only features used in training
    df = df[feature_names]
    
    return df


def interpret_score(score: float) -> tuple[str, str, str]:
    """
    Convert score to rating and recommendations.
    
    Returns: (rating, emoji, recommendation)
    """
    if score >= 80:
        return "Excellent", "🎣🌟", "Prime fishing conditions! Get out there!"
    elif score >= 65:
        return "Good", "🎣", "Favorable conditions, good chance of success"
    elif score >= 50:
        return "Fair", "🌊", "Moderate conditions, fishing possible"
    elif score >= 35:
        return "Poor", "⛅", "Challenging conditions, try another time"
    else:
        return "Very Poor", "⚠️", "Not recommended - wait for better conditions"


def print_prediction(date: dt.datetime, score: float, conditions: Dict):
    """Pretty print prediction results."""
    rating, emoji, recommendation = interpret_score(score)
    
    print("\n" + "="*60)
    print(f"🎣 FISHING FORECAST - {date.strftime('%A, %B %d, %Y')}")
    print("="*60)
    print(f"\n  Score: {score:.1f}/100  {emoji}")
    print(f"  Rating: {rating}")
    print(f"  {recommendation}")
    
    print("\n  Conditions:")
    temp_key = 'morning_temp_F' if 'morning_temp_F' in conditions else 'morning_temp_avg_F'
    if temp_key in conditions:
        print(f"    Water Temp: {conditions[temp_key]:.1f}°F")
    if 'pressure_mb' in conditions:
        print(f"    Pressure: {conditions['pressure_mb']:.1f} mb", end="")
        if 'pressure_change_6h' in conditions:
            change = conditions['pressure_change_6h']
            trend = "rising" if change > 0 else "falling" if change < 0 else "stable"
            print(f" ({trend})")
        else:
            print()
    
    print("\n" + "="*60 + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict optimal fishing conditions.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing trained model"
    )
    parser.add_argument(
        "--date",
        default="today",
        help="Date to predict (YYYY-MM-DD or 'today')"
    )
    parser.add_argument(
        "--forecast",
        type=int,
        help="Number of days to forecast (overrides --date)"
    )
    parser.add_argument(
        "--station",
        default="9413450",
        help="NOAA station ID (default: Monterey Harbor)"
    )
    
    args = parser.parse_args(argv)
    
    # Load model
    model_path = args.model_dir / 'fishing_model.joblib'
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Run train_fishing_model.py first!")
        return 1
    
    print(f"Loading model from {args.model_dir}...")
    model, scaler, metadata = load_model(args.model_dir)
    print(f"Model: {metadata['model_type']}")
    print(f"Training R²: {metadata['r2_score']:.4f}")
    
    feature_names = metadata['feature_names']
    
    # Determine dates to predict
    if args.forecast:
        dates = [dt.datetime.now() + dt.timedelta(days=i) for i in range(args.forecast)]
    else:
        if args.date.lower() == "today":
            dates = [dt.datetime.now()]
        else:
            dates = [dt.datetime.strptime(args.date, "%Y-%m-%d")]
    
    # Make predictions
    for pred_date in dates:
        # Fetch historical pattern-based conditions
        conditions = fetch_current_conditions(args.station, pred_date)
        conditions['hour'] = 7  # Morning fishing time
        conditions['day_of_week'] = pred_date.weekday()
        conditions['month'] = pred_date.month
        
        # Prepare features
        X = prepare_prediction_features(conditions, feature_names)
        
        # Scale if model requires it (check if scaler was used)
        try:
            X_scaled = scaler.transform(X)
            score = model.predict(X_scaled)[0]
        except:
            # If scaling fails, use unscaled (for tree-based models)
            score = model.predict(X)[0]
        
        # Clip to 0-100 range
        score = np.clip(score, 0, 100)
        
        # Print results
        print_prediction(pred_date, score, conditions)
    
    print("💡 NOTE: Predictions based on historical patterns (1993-2025).")
    print("   For live conditions, integrate real-time NOAA API.\n")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
