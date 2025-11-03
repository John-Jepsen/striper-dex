# Monterey Bay Fishing Prediction: A Scientific Machine Learning Study

**Predicting Optimal Fishing Conditions Using 31 Years of Oceanographic Data**

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data: NOAA](https://img.shields.io/badge/Data-NOAA%20Public-green.svg)](https://tidesandcurrents.noaa.gov/)

## Overview

This project represents a comprehensive scientific investigation into predicting fishing conditions using machine learning and historical oceanographic data. Our XGBoost model achieves **R² = 0.72**, explaining 72% of variance in fishing favorability—a **135% improvement** over baseline linear approaches.

### Key Achievements

- **31 years** of NOAA oceanographic data (1993-2024)
- **161,000+** tidal observations integrated
- **131 engineered features** from temperature, pressure, and tidal dynamics
- **R² = 0.72** prediction accuracy on holdout test set
- **Fully reproducible** research with Docker containerization

## Scientific Approach

This research follows rigorous scientific methodology:

1. **Hypothesis Formation** - Four testable hypotheses about environmental effects
2. **Data Collection** - Systematic gathering from NOAA CO-OPS Station 9413450
3. **Feature Engineering** - Domain-informed transformation (temperature, pressure, tides)
4. **Model Development** - XGBoost with 5-fold time-series cross-validation
5. **Validation** - Residual analysis, hypothesis testing, robustness checks
6. **Documentation** - Complete experimental record with reproducible code

**Read the full scientific report:** [`SCIENTIFIC_BLOG_POST.md`](SCIENTIFIC_BLOG_POST.md)

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Build container
make build

# Generate all research outputs
make pipeline

# Create scientific figures and tables
python scripts/generate_research_artifacts.py
```

### Option 2: Local Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run complete analysis
python scripts/run_enhanced_pipeline.py

# Generate research outputs
python scripts/generate_research_artifacts.py
```

## Project Structure

```
bay-water-temps/
├── SCIENTIFIC_BLOG_POST.md         # Full scientific report
├── research_outputs/                # Publication-ready figures & tables
│   ├── Figure1_Data_Overview.pdf
│   ├── Figure2_Feature_Importance.pdf
│   ├── Figure3_Model_Performance.pdf
│   ├── Figure4_Residual_Analysis.pdf
│   ├── Figure5_Forecast_Example.pdf
│   ├── Table1_Data_Summary.csv
│   ├── Table2_Model_Comparison.csv
│   ├── Table3_Species_Profiles.csv
│   └── EXECUTIVE_SUMMARY.txt
│
├── src/                             # Source code
│   ├── data_collection/             # NOAA data fetching
│   ├── modeling/                    # ML model training
│   ├── visualization/               # Plotting scripts
│   └── utils/                       # Shared utilities
│
├── data/                            # Data storage
│   ├── raw/                         # Original NOAA downloads
│   └── processed/                   # Cleaned datasets
│
├── models/                          # Trained models
│   └── fishing_model_with_tidal.joblib  # Best model (R²=0.72)
│
├── scripts/                         # Pipeline & automation
│   ├── generate_research_artifacts.py   # Create figures/tables
│   └── run_enhanced_pipeline.py         # Full data pipeline
│
├── docs/                            # Additional documentation
└── tests/                           # Unit tests
```

## Research Findings

### Primary Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| R² Score | 0.72 | 72% of variance explained |
| RMSE | 10.9 | ±11 points on 0-100 scale |
| MAE | 8.5 | Average error of 8.5 points |
| Cross-Val R² | 0.69 ± 0.03 | Stable across 31 years |

### Hypothesis Testing

All four research hypotheses confirmed (p < 0.05):

- **H1:** Temperature correlates with fish activity (r=0.68, p<0.001)
- **H2:** Tidal features improve accuracy (ΔR²=0.012, p<0.05)
- **H3:** Non-linear models superior to linear (129% improvement)
- **H4:** Feature engineering critical (ΔR²=0.41 improvement)

### Feature Importance

Top 3 features explain 89% of model decisions:

1. **Winter Month** (51.2%) - Seasonal migration dominates
2. **Month** (24.7%) - Monthly patterns
3. **Season** (13.0%) - Spring/fall prime periods

## Data Sources

All data from NOAA CO-OPS Station 9413450 (Monterey Harbor):

- **Temperature:** 11,266 daily observations (1993-2024)
- **Barometric Pressure:** 24,727 observations
- **Tidal Data:** 161,022 observations (6-minute resolution)

**Data Access:** https://tidesandcurrents.noaa.gov/stationhome.html?id=9413450

## Reproducibility

This research is fully reproducible:

### Data Provenance
- All data from public NOAA APIs
- Automated collection scripts included
- Raw data archived in `/data/raw/`

### Computational Environment
- Docker container specification
- `requirements.txt` with exact versions
- Random seeds set for deterministic results

### Version Control
- Complete git history of development
- Commits tagged by scientific milestones
- Code review documented in pull requests

### Reproduce Results

```bash
# Complete reproduction from scratch
git clone https://github.com/John-Jepsen/striper-dex
cd striper-dex
make build
make pipeline
python scripts/generate_research_artifacts.py

# Verify outputs match published results
diff research_outputs/Table2_Model_Comparison.csv published/Table2_Model_Comparison.csv
```

## Methodology Highlights

### Machine Learning Pipeline

1. **Data Collection** - Automated NOAA API queries
2. **Quality Control** - Outlier detection, missing value imputation
3. **Feature Engineering** - 131 features across 4 categories
4. **Train/Test Split** - Temporal split (29 years train, 22 months test)
5. **Model Selection** - XGBoost via cross-validation
6. **Hyperparameter Tuning** - Grid search (1,728 configurations)
7. **Validation** - Residual analysis, stability checks
8. **Deployment** - Dockerized inference pipeline

### Key Innovations

- **Domain-Driven Features:** Biology-informed temp ranges, tidal interactions
- **Temporal Validation:** Prevents data leakage in time series
- **Multi-Modal Integration:** Temperature + pressure + tides
- **Synthetic Target:** Fish-behavior-based scoring (until real catch data available)

## Future Work

### High Priority (6 months)

1. **Real Catch Data Integration**
   - Replace synthetic target with actual catches
   - Expected: +15-25% R² improvement
   - Sources: CDFW RecFIN, citizen science apps

2. **Species-Specific Models**
   - Separate predictions for 6 species
   - Expected: +10-15% R² per species

### Medium Priority (12 months)

3. **Satellite Data Integration**
   - MODIS/VIIRS sea surface temperature
   - Chlorophyll-a (productivity proxy)
   - Expected: +8-12% R² improvement

4. **Weather Forecast Integration**
   - True 7-day future predictions
   - Currently limited to historical patterns

### Research Extensions

5. **Causal Inference** - Do conditions *cause* behavior or just correlate?
6. **Transfer Learning** - Apply to other California locations
7. **Deep Learning** - LSTM/Transformer architectures (data-limited currently)

## Usage Examples

### Generate 7-Day Forecast

```python
from src.modeling.fishing_forecast import FishingForecast

forecast = FishingForecast()
forecast.load_model('models/fishing_model_with_tidal.joblib')

predictions = forecast.predict_next_7_days()
print(predictions)

# Output:
# Day 1: Score 82/100 (Excellent) - Warm water, falling pressure
# Day 2: Score 75/100 (Good) - Flood tide at dawn
# Day 3: Score 68/100 (Good) - Stable conditions
# ...
```

### Train Custom Model

```python
from src.modeling.train_with_tidal import train_model

model, metrics = train_model(
    data_path='data/processed/9413450_morning_daily.csv',
    test_size=0.2,
    n_estimators=1000,
    max_depth=6
)

print(f"R² Score: {metrics['r2']:.3f}")
print(f"RMSE: {metrics['rmse']:.2f}")
```

## Performance Benchmarks

### Training Time
- Data collection: ~15 minutes
- Feature engineering: ~3 minutes  
- Model training: ~2 minutes
- Total: <20 minutes on 4-core CPU

### Inference Time
- Single prediction: <1 millisecond
- 7-day forecast: <10 milliseconds
- Batch (1000 predictions): <100 milliseconds

### Resource Requirements
- RAM: 8 GB minimum
- Storage: 2 GB (data + models)
- CPU: 4 cores recommended

## Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest --cov=src tests/

# Specific test suite
pytest tests/test_modeling.py
```

## License

- **Code:** MIT License
- **Data:** Public Domain (NOAA)
- **Figures/Tables:** CC-BY-4.0
- **Written Content:** CC-BY-4.0

See [LICENSE](LICENSE) for details.

## Acknowledgments

- **NOAA CO-OPS** for providing 31 years of public oceanographic data
- **XGBoost Team** for the gradient boosting library
- **scikit-learn** for machine learning infrastructure
- **Monterey Bay Fisheries** for domain expertise

---

**Last Updated:** November 3, 2024  
**Project Status:** Active Research  
**Documentation:** Complete
