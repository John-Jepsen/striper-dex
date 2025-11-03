#!/usr/bin/env python3
"""
Visualize how the model responds to striped bass behavioral patterns.
Shows model predictions across different environmental conditions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

def load_model():
    """Load trained model and scaler."""
    model = joblib.load('models/fishing_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    return model, scaler

def create_test_scenario(temp_range, pressure_change, season='fall', hour=7):
    """Create test data for different scenarios."""
    scenarios = []
    
    for temp in temp_range:
        scenario = {
            'pressure_mb': 1015.0,
            'pressure_change_6h': pressure_change,
            'pressure_change_24h': pressure_change * 2,
            'temp_change_1d': 0,
            'temp_change_7d': 0,
            'temp_rolling_mean_7d': temp,
            'temp_rolling_std_7d': 1.0,
            'temp_anomaly_7d': 0,
            'temp_volatility_7d': 0.5,
            'pressure_stability_6h': 0.5,
            'hour': hour,
            'day_of_week': 5,  # Saturday
            'month': 10 if season == 'fall' else 4,
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'month_sin': np.sin(2 * np.pi * 10 / 12),
            'month_cos': np.cos(2 * np.pi * 10 / 12),
            'is_early_morning': 1 if 5 <= hour <= 9 else 0,
            'is_weekend': 1,
            'is_high_pressure': 0,
            'is_low_pressure': 0,
            'temp_in_optimal_range': 1 if 50 <= temp <= 58 else 0,
        }
        scenarios.append(scenario)
    
    return pd.DataFrame(scenarios)

def main():
    print("Loading model...")
    model, scaler = load_model()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Striped Bass Fishing Model - Behavioral Validation', 
                 fontsize=16, fontweight='bold')
    
    # 1. Temperature Response Curve
    print("Analyzing temperature response...")
    temps = np.linspace(45, 75, 50)
    
    # Stable pressure
    df_stable = create_test_scenario(temps, pressure_change=0)
    X_stable = scaler.transform(df_stable)
    scores_stable = model.predict(X_stable)
    
    # Falling pressure (pre-storm)
    df_falling = create_test_scenario(temps, pressure_change=-1.5)
    X_falling = scaler.transform(df_falling)
    scores_falling = model.predict(X_falling)
    
    ax1 = axes[0, 0]
    ax1.plot(temps, scores_stable, 'b-', linewidth=2, label='Stable Pressure')
    ax1.plot(temps, scores_falling, 'r-', linewidth=2, label='Falling Pressure (Pre-storm)')
    ax1.axvspan(60, 70, alpha=0.2, color='green', label='Optimal Range (60-70°F)')
    ax1.axvspan(55, 60, alpha=0.1, color='yellow', label='Good Range (55-60°F)')
    ax1.axvspan(50, 55, alpha=0.05, color='orange', label='Tolerable (50-55°F)')
    ax1.set_xlabel('Water Temperature (°F)', fontsize=11)
    ax1.set_ylabel('Fishing Quality Score', fontsize=11)
    ax1.set_title('Temperature Response (Striped Bass Preferences)', fontweight='bold')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # 2. Barometric Pressure Effect
    print("Analyzing pressure response...")
    pressure_changes = np.linspace(-3, 2, 50)
    scenarios_press = []
    
    for pc in pressure_changes:
        scenario = create_test_scenario([62], pressure_change=pc)[0:1]  # Optimal temp
        scenarios_press.append(scenario)
    
    df_press = pd.concat(scenarios_press, ignore_index=True)
    X_press = scaler.transform(df_press)
    scores_press = model.predict(X_press)
    
    ax2 = axes[0, 1]
    ax2.plot(pressure_changes, scores_press, 'purple', linewidth=2)
    ax2.axvspan(-3, -1.5, alpha=0.3, color='green', label='Rapidly Falling (Prime)')
    ax2.axvspan(-1.5, -0.5, alpha=0.2, color='lightgreen', label='Falling (Good)')
    ax2.axvspan(-0.5, 0.5, alpha=0.1, color='yellow', label='Stable')
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Pressure Change (mb/6h)', fontsize=11)
    ax2.set_ylabel('Fishing Quality Score', fontsize=11)
    ax2.set_title('Barometric Pressure Effect (at optimal temp 62°F)', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # 3. Time of Day Effect
    print("Analyzing time of day...")
    hours = np.arange(0, 24)
    scenarios_time = []
    
    for h in hours:
        scenario = create_test_scenario([62], pressure_change=-1.0, hour=h)[0:1]
        scenarios_time.append(scenario)
    
    df_time = pd.concat(scenarios_time, ignore_index=True)
    X_time = scaler.transform(df_time)
    scores_time = model.predict(X_time)
    
    ax3 = axes[1, 0]
    ax3.plot(hours, scores_time, 'orange', linewidth=2, marker='o', markersize=4)
    ax3.axvspan(5, 9, alpha=0.2, color='green', label='Early Morning (Prime)')
    ax3.axvspan(17, 20, alpha=0.15, color='lightgreen', label='Evening')
    ax3.set_xlabel('Hour of Day', fontsize=11)
    ax3.set_ylabel('Fishing Quality Score', fontsize=11)
    ax3.set_title('Time of Day Effect (Dawn/Dusk Feeding)', fontweight='bold')
    ax3.set_xticks(np.arange(0, 24, 3))
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    ax3.set_ylim(0, 100)
    
    # 4. Combined Scenario Matrix
    print("Creating scenario matrix...")
    temps_matrix = [50, 55, 60, 65, 70]
    pressure_matrix = [-2, -1, 0, 1]
    
    scenario_scores = np.zeros((len(pressure_matrix), len(temps_matrix)))
    
    for i, pc in enumerate(pressure_matrix):
        for j, temp in enumerate(temps_matrix):
            scenario = create_test_scenario([temp], pressure_change=pc)[0:1]
            X = scaler.transform(scenario)
            scenario_scores[i, j] = model.predict(X)[0]
    
    ax4 = axes[1, 1]
    im = ax4.imshow(scenario_scores, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax4.set_xticks(range(len(temps_matrix)))
    ax4.set_xticklabels([f'{t}°F' for t in temps_matrix])
    ax4.set_yticks(range(len(pressure_matrix)))
    ax4.set_yticklabels([f'{p:+.0f}' for p in pressure_matrix])
    ax4.set_xlabel('Water Temperature', fontsize=11)
    ax4.set_ylabel('Pressure Change (mb/6h)', fontsize=11)
    ax4.set_title('Optimal Condition Matrix', fontweight='bold')
    
    # Add text annotations
    for i in range(len(pressure_matrix)):
        for j in range(len(temps_matrix)):
            text = ax4.text(j, i, f'{scenario_scores[i, j]:.0f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Fishing Quality Score', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    output_path = Path('figures/model_behavior_validation.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved to {output_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
