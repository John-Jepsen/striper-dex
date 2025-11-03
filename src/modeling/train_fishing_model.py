#!/usr/bin/env python3
"""
Train ML models to predict optimal fishing conditions in Monterey Bay.

Models:
1. Multi-Linear Regression (baseline, interpretable)
2. Random Forest Regressor (non-linear patterns)
3. Gradient Boosting (XGBoost/LightGBM)
4. Time-series forecasters (Prophet, ARIMA)

Target Variable:
- Synthetic "fishing quality score" (0-100) based on known optimal conditions
- Later to be replaced with actual catch data when available
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import joblib

warnings.filterwarnings('ignore')


def create_synthetic_target(df: pd.DataFrame, use_optimized_scaling: bool = True) -> pd.Series:
    """
    Create synthetic fishing quality score based on STRIPED BASS biology.
    
    Based on striped bass behavioral science:
    - Water temp: Optimal 60-70°F (peak feeding), tolerate 55-70°F, sluggish <50°F
    - Spawning: 60-69°F peak activity
    - Barometric pressure: Falling = prime (pre-storm aggression), stable = good, high = slow
    - Tide: Moving water (flood/ebb) = active, slack = poor
    - Season: Spring/Fall migration periods are prime
    
    Score: 0 (poor) to 100 (excellent)
    
    Uses convex optimization to rescale scores so best Monterey Bay conditions
    approach 100, while acknowledging no day is perfect (max ~95).
    
    Sources: Striped bass behavioral research & fishing guides
    """
    score = pd.Series(50.0, index=df.index)  # Start at neutral 50
    
    # === TEMPERATURE (±35 points) - Most critical factor ===
    if 'morning_temp_avg_F' in df.columns:
        temp = df['morning_temp_avg_F']
        
        # Ideal range: 60-70°F (peak feeding)
        in_ideal = (temp >= 60) & (temp <= 70)
        score[in_ideal] += 35
        
        # Good range: 55-60°F (active but not peak)
        in_good = (temp >= 55) & (temp < 60)
        score[in_good] += 20
        
        # Tolerable: 50-55°F (will feed if bait present)
        in_tolerable = (temp >= 50) & (temp < 55)
        score[in_tolerable] += 10
        
        # Cold: <50°F (sluggish, poor fishing)
        in_cold = temp < 50
        score[in_cold] -= 20
        
        # Hot: >70°F (seeking deeper/cooler water)
        in_hot = temp > 70
        score[in_hot] -= 10
        
        # Spawning bonus: 61-69°F peak spawning activity
        in_spawn = (temp >= 61) & (temp <= 69)
        score[in_spawn] += 10
    
    # === BAROMETRIC PRESSURE (±25 points) - Triggers feeding ===
    if 'pressure_change_6h' in df.columns:
        pressure_change = df['pressure_change_6h'].fillna(0)
        
        # Falling pressure = PRIME TIME (pre-storm aggression)
        falling = pressure_change < -0.5
        score[falling] += 25
        
        # Rapidly falling = extremely aggressive
        rapid_fall = pressure_change < -1.5
        score[rapid_fall] += 10  # Extra bonus
        
        # Stable pressure = good baseline fishing
        stable = (pressure_change >= -0.5) & (pressure_change <= 0.5)
        score[stable] += 10
        
        # High/rising pressure = slower fishing
        rising = pressure_change > 0.5
        score[rising] -= 10
        
        # Very high pressure = lethargic fish
        if 'is_high_pressure' in df.columns:
            score[df['is_high_pressure'] == 1] -= 15
    
    # === TIDE PHASE (±15 points) - Moving water is key ===
    # Note: We don't have tidal data yet, but will add when available
    # Incoming/outgoing = active feeding
    # Slack = poor activity
    
    # === TIME OF DAY (±10 points) ===
    if 'is_early_morning' in df.columns:
        # Dawn/dusk are prime (early morning proxy)
        score += df['is_early_morning'] * 10
    
    # === SEASON (±10 points) - Migration periods ===
    if 'season' in df.columns:
        # Spring = incoming migration, Fall = outgoing migration
        season_scores = {
            'spring': 15,  # Prime - incoming migration
            'fall': 15,    # Prime - outgoing migration  
            'summer': 0,   # Neutral - resident fish, seeking cool water
            'winter': -10  # Poor - fish move to deeper water
        }
        score += df['season'].map(season_scores).fillna(0)
    
    # === TEMPERATURE STABILITY (±5 points) ===
    if 'temp_volatility_7d' in df.columns:
        volatility = df['temp_volatility_7d'].fillna(df['temp_volatility_7d'].median())
        # Stable temps = predictable feeding patterns
        stability_score = 5 * np.exp(-volatility / 2)
        score += stability_score - 2.5
    
    # === TEMPERATURE TREND (±5 points) ===
    if 'temp_change_7d' in df.columns:
        temp_trend = df['temp_change_7d'].fillna(0)
        # Warming in spring = fish moving in (positive)
        # Cooling in fall = fish still active (neutral to positive)
        if 'season' in df.columns:
            warming_spring = (temp_trend > 0) & (df['season'] == 'spring')
            score[warming_spring] += 5
            
            warming_fall = (temp_trend > 0) & (df['season'] == 'fall')
            score[warming_fall] -= 5  # Fish leaving
    
    # Add realistic variation (fish behavior isn't perfectly predictable)
    noise = np.random.normal(0, 3, size=len(score))
    score += noise
    
    # Clip to 0-100 range
    score = score.clip(0, 100)
    
    if use_optimized_scaling:
        # Use convex optimization to rescale based on Monterey Bay reality
        # Find the 99th percentile (best realistic conditions)
        # and scale so those approach 95 (acknowledging no day is perfect)
        p99 = score.quantile(0.99)
        p01 = score.quantile(0.01)
        
        # Rescale using convex transformation
        # Map [p01, p99] -> [5, 95] to spread the distribution
        score_rescaled = 5 + (score - p01) * 90 / (p99 - p01)
        score = score_rescaled.clip(0, 100)
        
        print(f"  Optimized scaling: {p01:.1f}-{p99:.1f} → 5-95")
        print(f"  Post-scaling range: {score.min():.1f}-{score.max():.1f}")
    
    return score


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select and prepare features for modeling.
    
    Returns:
        (feature_df, feature_names)
    """
    # Define feature categories
    numeric_features = [
        'morning_temp_avg_F',
        'pressure_mb',
        'pressure_change_6h',
        'pressure_change_24h',
        'temp_change_1d',
        'temp_change_7d',
        'temp_rolling_mean_7d',
        'temp_rolling_std_7d',
        'temp_anomaly_7d',
        'temp_volatility_7d',
        'pressure_stability_6h',
        'hour',
        'day_of_week',
        'month',
        'hour_sin',
        'hour_cos',
        'month_sin',
        'month_cos',
        'is_early_morning',
        'is_weekend',
        'is_high_pressure',
        'is_low_pressure',
        'temp_in_optimal_range',
    ]
    
    # Select only features that exist
    available_features = [f for f in numeric_features if f in df.columns]
    
    X = df[available_features].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    return X, available_features


def train_linear_model(X_train, y_train, X_test, y_test) -> Dict[str, Any]:
    """Train multi-linear regression model."""
    print("\n" + "="*60)
    print("MULTI-LINEAR REGRESSION")
    print("="*60)
    
    # Try different regularization
    models = {
        'OLS': LinearRegression(),
        'Ridge (L2)': Ridge(alpha=1.0),
        'Lasso (L1)': Lasso(alpha=0.1),
    }
    
    best_score = -np.inf
    best_model = None
    best_name = None
    
    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        # Train on full training set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"\n{name}:")
        print(f"  CV R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  Test R²: {r2:.4f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}")
        
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_name = name
    
    print(f"\n✓ Best: {best_name} (R² = {best_score:.4f})")
    
    return {
        'model': best_model,
        'name': best_name,
        'r2': best_score,
        'cv_scores': cv_scores.tolist(),
    }


def train_random_forest(X_train, y_train, X_test, y_test) -> Dict[str, Any]:
    """Train Random Forest model with hyperparameter tuning."""
    print("\n" + "="*60)
    print("RANDOM FOREST REGRESSOR")
    print("="*60)
    
    # Grid search for hyperparameters
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    print("Running grid search (this may take a minute)...")
    grid_search = GridSearchCV(
        rf, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Test R²: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    return {
        'model': best_model,
        'name': 'Random Forest',
        'r2': r2,
        'best_params': grid_search.best_params_,
    }


def train_gradient_boosting(X_train, y_train, X_test, y_test) -> Dict[str, Any]:
    """Train Gradient Boosting model."""
    print("\n" + "="*60)
    print("GRADIENT BOOSTING REGRESSOR")
    print("="*60)
    
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbose=0,
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Test R²: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    return {
        'model': model,
        'name': 'Gradient Boosting',
        'r2': r2,
    }


def plot_feature_importance(model, feature_names: List[str], output_dir: Path):
    """Plot feature importance for tree-based models."""
    if not hasattr(model, 'feature_importances_'):
        return
    
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:20]  # Top 20
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importance[indices], color='#2a9d8f')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top 20 Most Important Features')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=140)
    plt.close()
    print(f"  Saved feature importance plot")


def plot_predictions(y_test, y_pred, model_name: str, output_dir: Path):
    """Plot actual vs predicted values."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(y_test, y_pred, alpha=0.5, s=30)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    
    ax.set_xlabel('Actual Fishing Quality Score')
    ax.set_ylabel('Predicted Fishing Quality Score')
    ax.set_title(f'{model_name}: Predicted vs Actual')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    filename = model_name.lower().replace(' ', '_') + '_predictions.png'
    plt.savefig(output_dir / filename, dpi=140)
    plt.close()


def plot_residuals(y_test, y_pred, model_name: str, output_dir: Path):
    """Plot residual distribution."""
    residuals = y_test - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Residual histogram
    ax1.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Residuals')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Residual Distribution')
    ax1.axvline(0, color='red', linestyle='--', lw=2)
    
    # Residual vs predicted
    ax2.scatter(y_pred, residuals, alpha=0.5, s=30)
    ax2.axhline(0, color='red', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals vs Predicted')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    filename = model_name.lower().replace(' ', '_') + '_residuals.png'
    plt.savefig(output_dir / filename, dpi=140)
    plt.close()


def save_model(model, scaler, feature_names: List[str], metadata: Dict, output_dir: Path):
    """Save trained model and metadata."""
    model_path = output_dir / 'fishing_model.joblib'
    scaler_path = output_dir / 'scaler.joblib'
    metadata_path = output_dir / 'model_metadata.json'
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    metadata['feature_names'] = feature_names
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Model saved to {output_dir}/")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train fishing prediction models.")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/features/fishing_features.csv"),
        help="Feature CSV from feature_engineering.py"
    )
    parser.add_argument(
        "--model",
        choices=['mlr', 'rf', 'gbm', 'all'],
        default='all',
        help="Model to train (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save models and plots"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set proportion (default: 0.2)"
    )
    
    args = parser.parse_args(argv)
    
    # Load features
    if not args.features.exists():
        print(f"Error: Feature file not found: {args.features}")
        print("Run feature_engineering.py first!")
        return 1
    
    print(f"Loading features from {args.features}")
    df = pd.read_csv(args.features)
    print(f"Loaded {len(df)} samples")
    
    # Create synthetic target (TODO: replace with real catch data)
    print("\nCreating synthetic fishing quality scores...")
    print("(This will be replaced with actual catch data in production)")
    df['fishing_quality_score'] = create_synthetic_target(df)
    
    # Prepare features
    X, feature_names = prepare_features(df)
    y = df['fishing_quality_score']
    
    print(f"\nFeatures: {len(feature_names)}")
    print(f"Samples: {len(X)}")
    print(f"Target range: {y.min():.1f} - {y.max():.1f}")
    
    # Train/test split (temporal split for time series)
    split_idx = int(len(X) * (1 - args.test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train models
    results = {}
    
    if args.model in ['mlr', 'all']:
        results['mlr'] = train_linear_model(X_train_scaled, y_train, X_test_scaled, y_test)
        plot_predictions(y_test, results['mlr']['model'].predict(X_test_scaled), 
                        'Multi-Linear Regression', args.output_dir)
        plot_residuals(y_test, results['mlr']['model'].predict(X_test_scaled),
                      'Multi-Linear Regression', args.output_dir)
    
    if args.model in ['rf', 'all']:
        results['rf'] = train_random_forest(X_train, y_train, X_test, y_test)
        plot_feature_importance(results['rf']['model'], feature_names, args.output_dir)
        plot_predictions(y_test, results['rf']['model'].predict(X_test),
                        'Random Forest', args.output_dir)
        plot_residuals(y_test, results['rf']['model'].predict(X_test),
                      'Random Forest', args.output_dir)
    
    if args.model in ['gbm', 'all']:
        results['gbm'] = train_gradient_boosting(X_train, y_train, X_test, y_test)
        plot_feature_importance(results['gbm']['model'], feature_names, args.output_dir)
        plot_predictions(y_test, results['gbm']['model'].predict(X_test),
                        'Gradient Boosting', args.output_dir)
        plot_residuals(y_test, results['gbm']['model'].predict(X_test),
                      'Gradient Boosting', args.output_dir)
    
    # Select best model
    best_model_key = max(results, key=lambda k: results[k]['r2'])
    best_model_info = results[best_model_key]
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for key, info in results.items():
        print(f"{info['name']}: R² = {info['r2']:.4f}")
    print(f"\n🏆 Best model: {best_model_info['name']} (R² = {best_model_info['r2']:.4f})")
    
    # Save best model
    metadata = {
        'model_type': best_model_info['name'],
        'r2_score': best_model_info['r2'],
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'trained_date': pd.Timestamp.now().isoformat(),
    }
    
    save_model(best_model_info['model'], scaler, feature_names, metadata, args.output_dir)
    
    print("\n✅ Training complete!")
    print(f"   Models and plots saved to {args.output_dir}/")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
