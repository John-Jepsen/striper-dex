#!/usr/bin/env python3
"""
Hypothesis testing and statistical analysis for fishing predictions.

Tests:
1. Correlation analysis: Do temp/pressure correlate with fishing quality?
2. ANOVA: Does fishing quality differ across temperature quartiles?
3. Chi-square: Are optimal conditions independent of season?
4. T-tests: Do rising vs falling pressure periods differ in quality?
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, chi2_contingency, f_oneway, ttest_ind

warnings.filterwarnings('ignore')


def load_data(path: Path) -> pd.DataFrame:
    """Load feature dataset with fishing scores."""
    df = pd.read_csv(path)
    return df


def test_correlation(df: pd.DataFrame, target_col: str = 'fishing_quality_score') -> Dict:
    """
    H0: No correlation between environmental variables and fishing quality
    H1: Significant correlation exists
    """
    print("\n" + "="*70)
    print("TEST 1: CORRELATION ANALYSIS (Pearson)")
    print("="*70)
    print("H0: Environmental variables are not correlated with fishing quality")
    print("H1: Significant correlation exists (α = 0.05)")
    print()
    
    variables = [
        'morning_temp_avg_F',
        'pressure_mb',
        'pressure_change_6h',
        'temp_change_1d',
        'temp_volatility_7d',
    ]
    
    results = {}
    
    for var in variables:
        if var not in df.columns or target_col not in df.columns:
            continue
        
        # Remove NaN values
        clean_data = df[[var, target_col]].dropna()
        
        if len(clean_data) < 10:
            continue
        
        r, p_value = pearsonr(clean_data[var], clean_data[target_col])
        
        # Interpret
        significance = "✓ SIGNIFICANT" if p_value < 0.05 else "✗ Not significant"
        strength = "weak"
        if abs(r) > 0.7:
            strength = "strong"
        elif abs(r) > 0.4:
            strength = "moderate"
        
        print(f"{var:30s} r={r:7.4f}  p={p_value:.4f}  {significance}")
        print(f"{'':30s} Strength: {strength} {'positive' if r > 0 else 'negative'} correlation")
        print()
        
        results[var] = {
            'r': r,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'strength': strength,
        }
    
    return results


def test_temperature_quartiles(df: pd.DataFrame, target_col: str = 'fishing_quality_score') -> Dict:
    """
    H0: Mean fishing quality is equal across temperature quartiles
    H1: At least one quartile has different mean fishing quality (ANOVA)
    """
    print("\n" + "="*70)
    print("TEST 2: ANOVA - Fishing Quality Across Temperature Quartiles")
    print("="*70)
    print("H0: Mean fishing quality is equal across all temperature ranges")
    print("H1: At least one temperature range has different fishing quality (α = 0.05)")
    print()
    
    temp_col = 'morning_temp_avg_F'
    
    if temp_col not in df.columns or target_col not in df.columns:
        print("Required columns not found!")
        return {}
    
    # Create quartiles
    df_clean = df[[temp_col, target_col]].dropna()
    df_clean['temp_quartile'] = pd.qcut(df_clean[temp_col], q=4, labels=['Q1 (coldest)', 'Q2', 'Q3', 'Q4 (warmest)'])
    
    # Group by quartile
    groups = [group[target_col].values for name, group in df_clean.groupby('temp_quartile')]
    
    # ANOVA
    f_stat, p_value = f_oneway(*groups)
    
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("✓ REJECT H0: Temperature significantly affects fishing quality!")
    else:
        print("✗ FAIL TO REJECT H0: No significant temperature effect")
    
    print("\nQuartile Statistics:")
    for name, group in df_clean.groupby('temp_quartile'):
        temp_range = group[temp_col]
        quality = group[target_col]
        print(f"  {name}: {temp_range.min():.1f}-{temp_range.max():.1f}°F  →  "
              f"Quality: {quality.mean():.1f} ± {quality.std():.1f}")
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'quartile_means': {str(name): group[target_col].mean() 
                          for name, group in df_clean.groupby('temp_quartile')},
    }


def test_pressure_trend(df: pd.DataFrame, target_col: str = 'fishing_quality_score') -> Dict:
    """
    H0: Fishing quality is equal for rising vs falling pressure
    H1: Rising and falling pressure have different fishing quality (t-test)
    """
    print("\n" + "="*70)
    print("TEST 3: T-TEST - Rising vs Falling Barometric Pressure")
    print("="*70)
    print("H0: Fishing quality is equal for rising and falling pressure periods")
    print("H1: Fishing quality differs between pressure trends (α = 0.05)")
    print()
    
    pressure_change_col = 'pressure_change_6h'
    
    if pressure_change_col not in df.columns or target_col not in df.columns:
        print("Required columns not found!")
        return {}
    
    df_clean = df[[pressure_change_col, target_col]].dropna()
    
    # Classify as rising (>0.5 mb) or falling (<-0.5 mb)
    rising = df_clean[df_clean[pressure_change_col] > 0.5][target_col]
    falling = df_clean[df_clean[pressure_change_col] < -0.5][target_col]
    
    if len(rising) < 5 or len(falling) < 5:
        print("Insufficient data for test!")
        return {}
    
    t_stat, p_value = ttest_ind(rising, falling)
    
    print(f"Rising pressure periods: n={len(rising)}, mean quality={rising.mean():.1f} ± {rising.std():.1f}")
    print(f"Falling pressure periods: n={len(falling)}, mean quality={falling.mean():.1f} ± {falling.std():.1f}")
    print()
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        winner = "rising" if rising.mean() > falling.mean() else "falling"
        print(f"✓ REJECT H0: {winner.upper()} pressure has significantly better fishing!")
    else:
        print("✗ FAIL TO REJECT H0: No significant difference")
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'rising_mean': rising.mean(),
        'falling_mean': falling.mean(),
    }


def test_seasonal_independence(df: pd.DataFrame) -> Dict:
    """
    H0: Optimal fishing conditions are independent of season
    H1: Optimal conditions are associated with specific seasons (chi-square)
    """
    print("\n" + "="*70)
    print("TEST 4: CHI-SQUARE - Seasonal Independence")
    print("="*70)
    print("H0: Optimal fishing conditions occur independently of season")
    print("H1: Certain seasons have more optimal conditions (α = 0.05)")
    print()
    
    if 'season' not in df.columns or 'fishing_quality_score' not in df.columns:
        print("Required columns not found!")
        return {}
    
    df_clean = df[['season', 'fishing_quality_score']].dropna()
    
    # Define "optimal" as top 25% of scores
    threshold = df_clean['fishing_quality_score'].quantile(0.75)
    df_clean['is_optimal'] = (df_clean['fishing_quality_score'] >= threshold).astype(int)
    
    # Contingency table
    contingency = pd.crosstab(df_clean['season'], df_clean['is_optimal'])
    
    print("Contingency Table:")
    print(contingency)
    print()
    
    # Chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("✓ REJECT H0: Optimal conditions are season-dependent!")
        
        # Show which seasons are best
        optimal_by_season = df_clean.groupby('season')['is_optimal'].mean().sort_values(ascending=False)
        print("\nOptimal condition rate by season:")
        for season, rate in optimal_by_season.items():
            print(f"  {season:10s} {rate*100:.1f}%")
    else:
        print("✗ FAIL TO REJECT H0: Seasons are independent of optimal conditions")
    
    return {
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof,
        'significant': p_value < 0.05,
    }


def plot_correlation_matrix(df: pd.DataFrame, output_dir: Path):
    """Plot correlation heatmap of key variables."""
    variables = [
        'fishing_quality_score',
        'morning_temp_avg_F',
        'pressure_mb',
        'pressure_change_6h',
        'temp_change_1d',
        'temp_volatility_7d',
    ]
    
    available = [v for v in variables if v in df.columns]
    
    if len(available) < 3:
        return
    
    corr_matrix = df[available].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Correlation Matrix: Environmental Variables vs Fishing Quality', 
                 fontsize=14, weight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png', dpi=140, bbox_inches='tight')
    plt.close()
    print("\n  Saved correlation_matrix.png")


def plot_temperature_quality(df: pd.DataFrame, output_dir: Path):
    """Scatter plot of temperature vs fishing quality."""
    if 'morning_temp_avg_F' not in df.columns or 'fishing_quality_score' not in df.columns:
        return
    
    df_clean = df[['morning_temp_avg_F', 'fishing_quality_score']].dropna()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter
    ax.scatter(df_clean['morning_temp_avg_F'], df_clean['fishing_quality_score'],
               alpha=0.4, s=40, color='#2a9d8f')
    
    # Trend line
    z = np.polyfit(df_clean['morning_temp_avg_F'], df_clean['fishing_quality_score'], 2)
    p = np.poly1d(z)
    x_trend = np.linspace(df_clean['morning_temp_avg_F'].min(), 
                         df_clean['morning_temp_avg_F'].max(), 100)
    ax.plot(x_trend, p(x_trend), 'r--', linewidth=2, alpha=0.8, label='Polynomial fit')
    
    # Optimal range shading
    ax.axvspan(50, 58, alpha=0.2, color='green', label='Optimal range (50-58°F)')
    
    ax.set_xlabel('Water Temperature (°F)', fontsize=12)
    ax.set_ylabel('Fishing Quality Score', fontsize=12)
    ax.set_title('Water Temperature vs Fishing Quality', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'temp_vs_quality.png', dpi=140, bbox_inches='tight')
    plt.close()
    print("  Saved temp_vs_quality.png")


def plot_seasonal_boxplot(df: pd.DataFrame, output_dir: Path):
    """Boxplot of fishing quality by season."""
    if 'season' not in df.columns or 'fishing_quality_score' not in df.columns:
        return
    
    df_clean = df[['season', 'fishing_quality_score']].dropna()
    
    season_order = ['winter', 'spring', 'summer', 'fall']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df_clean, x='season', y='fishing_quality_score', 
                order=season_order, palette='Set2', ax=ax)
    
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Fishing Quality Score', fontsize=12)
    ax.set_title('Fishing Quality Distribution by Season', fontsize=14, weight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'seasonal_quality.png', dpi=140, bbox_inches='tight')
    plt.close()
    print("  Saved seasonal_quality.png")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Statistical hypothesis testing for fishing predictions.")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/features/fishing_features.csv"),
        help="Feature CSV with fishing_quality_score"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Output directory for plots and results"
    )
    
    args = parser.parse_args(argv)
    
    if not args.features.exists():
        print(f"Error: {args.features} not found!")
        print("Run feature_engineering.py and train_fishing_model.py first!")
        return 1
    
    print("Loading data...")
    df = pd.read_csv(args.features)
    
    # If no fishing score, create synthetic one
    if 'fishing_quality_score' not in df.columns:
        print("Creating synthetic fishing quality scores...")
        from train_fishing_model import create_synthetic_target
        df['fishing_quality_score'] = create_synthetic_target(df)
    
    print(f"Loaded {len(df)} samples")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run statistical tests
    print("\n" + "="*70)
    print("HYPOTHESIS TESTING SUITE")
    print("="*70)
    
    correlation_results = test_correlation(df)
    anova_results = test_temperature_quartiles(df)
    ttest_results = test_pressure_trend(df)
    chi2_results = test_seasonal_independence(df)
    
    # Generate plots
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    plot_correlation_matrix(df, args.output_dir)
    plot_temperature_quality(df, args.output_dir)
    plot_seasonal_boxplot(df, args.output_dir)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY OF FINDINGS")
    print("="*70)
    
    significant_correlations = sum(1 for r in correlation_results.values() if r['significant'])
    print(f"\n✓ {significant_correlations}/{len(correlation_results)} variables show significant correlation")
    
    if anova_results.get('significant'):
        print("✓ Temperature range significantly affects fishing quality (ANOVA)")
    
    if ttest_results.get('significant'):
        winner = "Rising" if ttest_results['rising_mean'] > ttest_results['falling_mean'] else "Falling"
        print(f"✓ {winner} pressure periods have better fishing quality (t-test)")
    
    if chi2_results.get('significant'):
        print("✓ Optimal fishing conditions are season-dependent (χ²)")
    
    print(f"\n📊 Visualizations saved to {args.output_dir}/")
    print("\n✅ Hypothesis testing complete!")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
