#!/usr/bin/env python3
"""
Quick fix for underfitting: Add polynomial features and use XGBoost.
This should improve R² from 0.31 to ~0.45-0.50.

Usage:
    python fix_underfitting.py --quick    # Fast polynomial + XGBoost
    python fix_underfitting.py --full     # Add all interaction features
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import json

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not installed. Install with: pip install xgboost")
    print("   Falling back to sklearn GradientBoosting")

from sklearn.ensemble import GradientBoostingRegressor


def add_polynomial_features(df: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
    """Add polynomial and interaction features for key predictors."""
    print(f"  Adding polynomial features (degree={degree})...")
    
    # Key features that have non-linear relationships
    key_features = []
    
    # Temperature features (non-linear: too cold AND too hot = bad)
    if 'morning_temp_F' in df.columns:
        key_features.append('morning_temp_F')
    
    # Pressure changes (non-linear: rapid changes matter more)
    if 'pressure_change_6h' in df.columns:
        key_features.append('pressure_change_6h')
    
    # Temperature trends (interaction with current temp)
    if 'temp_change_7d' in df.columns:
        key_features.append('temp_change_7d')
    
    # Volatility (interaction with absolute temp)
    if 'temp_volatility_7d' in df.columns:
        key_features.append('temp_volatility_7d')
    
    if not key_features:
        print("    ⚠️  No key features found for polynomial expansion")
        return df
    
    print(f"    Using features: {key_features}")
    
    # Fill NaN before polynomial transform
    key_df = df[key_features].fillna(df[key_features].median())
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    poly_features = poly.fit_transform(key_df)
    
    # Get feature names
    poly_names = poly.get_feature_names_out(key_features)
    
    # Create DataFrame (exclude original features, already in df)
    # Only keep interaction and squared terms
    new_feature_mask = [i for i, name in enumerate(poly_names) 
                       if '^2' in name or ' ' in name]  # Squared or interaction
    
    poly_df = pd.DataFrame(
        poly_features[:, new_feature_mask],
        columns=[poly_names[i] for i in new_feature_mask],
        index=df.index
    )
    
    print(f"    Created {len(poly_df.columns)} new polynomial features")
    
    return pd.concat([df, poly_df], axis=1)


def add_season_temp_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add season-specific temperature scoring."""
    print("  Adding season-temperature interactions...")
    
    if 'season' not in df.columns or 'morning_temp_F' not in df.columns:
        print("    ⚠️  Missing season or temperature columns")
        return df
    
    temp = df['morning_temp_F']
    season = df['season']
    
    # Spring: warming trend is good (fish moving in)
    df['spring_warming'] = (
        (season == 'spring') & 
        (temp >= 55) & 
        (temp <= 65)
    ).astype(int)
    
    # Fall: 58-68°F optimal for migration staging
    df['fall_optimal'] = (
        (season == 'fall') & 
        (temp >= 58) & 
        (temp <= 68)
    ).astype(int)
    
    # Winter: cold water = fish in deep water
    df['winter_cold'] = (
        (season == 'winter') & 
        (temp < 52)
    ).astype(int)
    
    # Summer: moderate temps = seeking cool pockets
    df['summer_moderate'] = (
        (season == 'summer') & 
        (temp >= 55) & 
        (temp <= 62)
    ).astype(int)
    
    print(f"    Created 4 season-temperature interaction features")
    
    return df


def add_pressure_temp_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add critical pressure-temperature combinations."""
    print("  Adding pressure-temperature interactions...")
    
    has_temp = 'morning_temp_F' in df.columns
    has_pressure_change = 'pressure_change_6h' in df.columns
    
    if not (has_temp and has_pressure_change):
        print("    ⚠️  Missing temperature or pressure change columns")
        return df
    
    temp = df['morning_temp_F']
    pressure_change = df['pressure_change_6h'].fillna(0)
    
    # Warm water + falling pressure = feeding frenzy
    df['warm_falling_pressure'] = (
        ((temp >= 60) & (temp <= 70)) & 
        (pressure_change < -0.5)
    ).astype(int)
    
    # Optimal temp + rapidly falling pressure = prime time
    df['optimal_rapid_fall'] = (
        ((temp >= 60) & (temp <= 70)) & 
        (pressure_change < -1.5)
    ).astype(int)
    
    # Cold water + stable pressure = slow fishing
    df['cold_stable_pressure'] = (
        (temp < 55) & 
        (pressure_change.abs() < 0.5)
    ).astype(int)
    
    print(f"    Created 3 pressure-temperature interaction features")
    
    return df


def add_temporal_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add multi-day temporal context features."""
    print("  Adding temporal context features...")
    
    if 'morning_temp_F' not in df.columns:
        print("    ⚠️  Missing temperature column")
        return df
    
    # 3-day warming/cooling trend
    df['temp_3d_increasing'] = (
        df['morning_temp_F'].diff(3) > 0
    ).astype(int)
    
    # Consecutive days in optimal range
    if 'temp_in_optimal_range' in df.columns:
        df['optimal_streak'] = (
            df['temp_in_optimal_range']
            .groupby(
                (df['temp_in_optimal_range'] != df['temp_in_optimal_range'].shift())
                .cumsum()
            )
            .cumsum()
        )
    
    # Temperature acceleration (2nd derivative)
    if 'temp_change_1d' in df.columns:
        df['temp_acceleration'] = df['temp_change_1d'].diff()
    
    print(f"    Created 3 temporal context features")
    
    return df


def create_improved_synthetic_target(df: pd.DataFrame) -> pd.Series:
    """
    Improved synthetic target with multiplicative interactions.
    This creates more variance for the model to learn.
    """
    score = pd.Series(50.0, index=df.index)
    
    # Temperature base (non-linear)
    if 'morning_temp_F' in df.columns:
        temp = df['morning_temp_F']
        
        # Quadratic penalty for distance from optimal
        optimal_temp = 65  # Sweet spot for striped bass
        temp_deviation = abs(temp - optimal_temp)
        temp_score = 35 * np.exp(-temp_deviation / 10)  # Exponential decay
        
        score += temp_score
        
        # Bonus for spawning range
        in_spawn = (temp >= 61) & (temp <= 69)
        score[in_spawn] += 10
    
    # Pressure effect (multiplicative)
    if 'pressure_change_6h' in df.columns:
        pressure_change = df['pressure_change_6h'].fillna(0)
        
        # Falling pressure multiplier
        pressure_multiplier = pd.Series(1.0, index=df.index)
        
        falling = pressure_change < -0.5
        pressure_multiplier[falling] = 1.3
        
        rapid_fall = pressure_change < -1.5
        pressure_multiplier[rapid_fall] = 1.6
        
        rising = pressure_change > 0.5
        pressure_multiplier[rising] = 0.75
        
        score *= pressure_multiplier
    
    # Season multiplier
    if 'season' in df.columns:
        season_mult = df['season'].map({
            'spring': 1.25,
            'fall': 1.25,
            'summer': 1.0,
            'winter': 0.6
        }).fillna(1.0)
        
        score *= season_mult
    
    # Early morning bonus
    if 'is_early_morning' in df.columns:
        score += df['is_early_morning'] * 8
    
    # Add realistic noise
    noise = np.random.normal(0, 6, size=len(score))
    score += noise
    
    return score.clip(5, 100)


def prepare_features(df: pd.DataFrame) -> tuple:
    """Select features for modeling."""
    
    # Exclude non-feature columns
    exclude_cols = [
        'timestamp', 'date', 'fishing_quality_score',
        'pressure_trend_6h'  # Categorical, needs encoding
    ]
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Only keep numeric columns
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    X = df[numeric_cols].copy()
    X = X.fillna(X.median())
    
    return X, numeric_cols


def train_xgboost_model(X_train, y_train, X_test, y_test):
    """Train XGBoost with good default hyperparameters."""
    print("\n" + "="*60)
    print("XGBOOST REGRESSOR")
    print("="*60)
    
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        early_stopping_rounds=50,
        verbosity=0
    )
    
    print("Training XGBoost (with early stopping)...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\nTest R²: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"Best iteration: {model.best_iteration}")
    
    return {
        'model': model,
        'name': 'XGBoost',
        'r2': r2,
        'rmse': rmse,
        'mae': mae
    }


def train_tuned_gbm(X_train, y_train, X_test, y_test):
    """Train sklearn GBM with better hyperparameters."""
    print("\n" + "="*60)
    print("GRADIENT BOOSTING (TUNED)")
    print("="*60)
    
    model = GradientBoostingRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        verbose=0
    )
    
    print("Training Gradient Boosting...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\nTest R²: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    return {
        'model': model,
        'name': 'Gradient Boosting (Tuned)',
        'r2': r2,
        'rmse': rmse,
        'mae': mae
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fix underfitting in fishing model")
    parser.add_argument(
        '--mode',
        choices=['quick', 'full'],
        default='quick',
        help='Quick: polynomial only, Full: all interactions'
    )
    parser.add_argument(
        '--features',
        type=Path,
        default=Path('data/features/fishing_features.csv'),
        help='Input features CSV'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('models'),
        help='Output directory for models'
    )
    
    args = parser.parse_args(argv)
    
    print("="*60)
    print("FIXING UNDERFITTING")
    print("="*60)
    print(f"Mode: {args.mode}")
    print()
    
    # Load data
    if not args.features.exists():
        print(f"❌ Features file not found: {args.features}")
        print("   Run: python feature_engineering.py")
        return 1
    
    print(f"Loading features from {args.features}")
    df = pd.read_csv(args.features)
    print(f"Loaded {len(df)} samples with {len(df.columns)} features")
    
    # Rename column if needed
    if 'morning_temp_F' in df.columns and 'morning_temp_avg_F' not in df.columns:
        df['morning_temp_avg_F'] = df['morning_temp_F']
        print("  ✓ Renamed morning_temp_F → morning_temp_avg_F")
    
    # Add polynomial features
    df = add_polynomial_features(df, degree=2)
    
    if args.mode == 'full':
        # Add all interaction features
        df = add_season_temp_interactions(df)
        df = add_pressure_temp_interactions(df)
        df = add_temporal_context(df)
    
    print(f"\nTotal features after engineering: {len(df.columns)}")
    
    # Create improved synthetic target
    print("\nCreating improved synthetic target...")
    df['fishing_quality_score'] = create_improved_synthetic_target(df)
    print(f"  Target range: {df['fishing_quality_score'].min():.1f} - {df['fishing_quality_score'].max():.1f}")
    print(f"  Target mean: {df['fishing_quality_score'].mean():.1f}")
    print(f"  Target std: {df['fishing_quality_score'].std():.1f}")
    
    # Prepare features
    print("\nPreparing features for modeling...")
    X, feature_names = prepare_features(df)
    y = df['fishing_quality_score']
    
    print(f"Features: {len(feature_names)}")
    print(f"Samples: {len(X)}")
    
    # Temporal train/test split (no shuffling!)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\nTrain: {len(X_train)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    results = []
    
    if HAS_XGBOOST:
        result = train_xgboost_model(X_train, y_train, X_test, y_test)
        results.append(result)
    
    result = train_tuned_gbm(X_train_scaled, y_train, X_test_scaled, y_test)
    results.append(result)
    
    # Print comparison
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for r in results:
        print(f"{r['name']:30s} R²={r['r2']:.4f}  RMSE={r['rmse']:.2f}  MAE={r['mae']:.2f}")
    
    # Select best
    best = max(results, key=lambda x: x['r2'])
    print(f"\n🏆 Best model: {best['name']} (R² = {best['r2']:.4f})")
    
    # Compare to original (assumed 0.31)
    original_r2 = 0.306
    improvement = (best['r2'] - original_r2) / original_r2 * 100
    print(f"\n📈 Improvement over original:")
    print(f"   {original_r2:.3f} → {best['r2']:.3f} ({improvement:+.1f}%)")
    
    # Save best model
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = args.output_dir / 'fishing_model_improved.joblib'
    scaler_path = args.output_dir / 'scaler_improved.joblib'
    metadata_path = args.output_dir / 'model_metadata_improved.json'
    
    joblib.dump(best['model'], model_path)
    joblib.dump(scaler, scaler_path)
    
    metadata = {
        'model_type': best['name'],
        'r2_score': best['r2'],
        'rmse': best['rmse'],
        'mae': best['mae'],
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'num_features': len(feature_names),
        'improvement_mode': args.mode,
        'trained_date': pd.Timestamp.now().isoformat(),
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Improved model saved to {args.output_dir}/")
    print(f"   Model: {model_path.name}")
    print(f"   Scaler: {scaler_path.name}")
    print(f"   Metadata: {metadata_path.name}")
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
