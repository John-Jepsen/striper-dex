# Research Outputs

This directory contains all scientific artifacts supporting the publication:

**"Predicting Optimal Fishing Conditions in Monterey Bay: A Machine Learning Approach"**

## Contents

### Executive Summary
- **EXECUTIVE_SUMMARY.txt** - High-level overview of research findings

### Figures (PNG & PDF formats)

**Figure 1: Data Overview**
- Panel A: 31-year sea surface temperature time series
- Panel B: Seasonal temperature climatology
- Panel C: Barometric pressure record
- Shows data quality and temporal coverage

**Figure 2: Feature Importance**
- Panel A: Top 15 most important features
- Panel B: Importance by feature category
- Demonstrates dominance of seasonal patterns

**Figure 3: Model Performance**
- Panel A: R² progression across model versions
- Panel B: Error metrics comparison (RMSE vs MAE)
- Panel C: 5-fold cross-validation stability
- Panel D: Predicted vs actual scatter plot
- Validates model accuracy and robustness

**Figure 4: Residual Analysis**
- Panel A: Residual distribution (normality test)
- Panel B: Residuals vs predicted values (homoscedasticity)
- Panel C: Q-Q plot (normality assessment)
- Panel D: Residuals over time (temporal independence)
- Diagnostic plots for model assumptions

**Figure 5: Forecast Example**
- Panel A: 7-day temperature forecast with confidence intervals
- Panel B: Fishing favorability scores with quality zones
- Example of operational forecast output

### Tables (CSV format)

**Table 1: Data Summary**
- Data sources, temporal coverage, record counts
- Demonstrates comprehensive environmental monitoring

**Table 2: Model Comparison**  
- Performance metrics across model versions
- Shows 135% improvement from feature engineering + ML

**Table 3: Species Profiles**
- Six target species with optimal conditions
- Biological context for model development

## Usage

### Viewing Figures
PDF files are publication-ready (300 DPI). PNG files suitable for presentations.

### Importing Tables
```python
import pandas as pd

# Load data summary
data_summary = pd.read_csv('research_outputs/Table1_Data_Summary.csv')

# Load model comparison
model_comp = pd.read_csv('research_outputs/Table2_Model_Comparison.csv')

# Load species profiles
species = pd.read_csv('research_outputs/Table3_Species_Profiles.csv')
```

### Regenerating Artifacts
All figures and tables can be regenerated from raw data:

```bash
python scripts/generate_research_artifacts.py
```

This ensures full reproducibility of research outputs.

## Citation

If using these materials, please cite:

```
[Author Name] (2024). Predicting Optimal Fishing Conditions in Monterey Bay: 
A Machine Learning Approach. [Institution/Publication Details].
```

## License

Figures and tables: CC-BY-4.0 (Creative Commons Attribution)  
Data: Public Domain (NOAA)  
Code: MIT License

## Contact

For questions about research methodology or data access:
- Email: [contact information]
- GitHub: [repository URL]
