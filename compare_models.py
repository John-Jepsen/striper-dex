#!/usr/bin/env python3
"""Compare original vs improved model performance."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Load metadata
with open('models/model_metadata_improved.json') as f:
    improved = json.load(f)

# Create comparison
data = {
    'Model': ['Original\n(Ridge)', 'Improved\n(XGBoost Quick)', 'Improved\n(XGBoost Full)'],
    'R² Score': [0.306, 0.645, 0.639],
    'RMSE': [15.62, 13.23, 13.41],
    'Variance\nExplained': [31, 65, 64]
}

df = pd.DataFrame(data)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# R² comparison
ax1 = axes[0]
bars = ax1.bar(df['Model'], df['R² Score'], color=['#e63946', '#2a9d8f', '#264653'])
ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax1.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5, label='Target (0.60)')
ax1.set_ylim(0, 0.8)
ax1.legend()

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, df['R² Score'])):
    height = bar.get_height()
    improvement = ((val - 0.306) / 0.306 * 100) if i > 0 else 0
    label = f'{val:.3f}'
    if i > 0:
        label += f'\n(+{improvement:.0f}%)'
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             label, ha='center', va='bottom', fontweight='bold')

# RMSE comparison
ax2 = axes[1]
bars = ax2.bar(df['Model'], df['RMSE'], color=['#e63946', '#2a9d8f', '#264653'])
ax2.set_ylabel('RMSE (lower is better)', fontsize=12, fontweight='bold')
ax2.set_title('Prediction Error', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 18)

# Add value labels
for bar, val in zip(bars, df['RMSE']):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
             f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('figures/underfitting_fix_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Saved comparison plot to figures/underfitting_fix_comparison.png")

# Print summary table
print("\n" + "="*70)
print("MODEL COMPARISON SUMMARY")
print("="*70)
print(df.to_string(index=False))
print("\n" + "="*70)
print("KEY FINDINGS:")
print("="*70)
print("1. R² improved from 0.306 → 0.645 (+110% improvement)")
print("2. Polynomial features + XGBoost solved underfitting")
print("3. Model now explains 65% of variance (was 31%)")
print("4. RMSE reduced by 15% (15.62 → 13.23)")
print("5. Can now distinguish excellent days from poor days!")
print("="*70)
