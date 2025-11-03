# Monterey Bay Water Temperature & Fishing Forecast

Machine learning-powered fishing forecast system using 30+ years of NOAA oceanographic data.

## 🎣 Quick Start

### Local (Virtual Environment)
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the forecast
python forecast.py
```

### Docker (Recommended for Production)
```bash
# Build the containers
make build

# Run fishing forecast
make forecast

# Or run the full pipeline
make pipeline
```

## 📁 Project Structure

```
bay-water-temps/
├── src/
│   ├── data_collection/     # NOAA data fetching scripts
│   │   ├── pull_noaa_history.py
│   │   ├── buoy-temps.py
│   │   ├── collect_weather_data.py
│   │   ├── collect_tidal_data.py
│   │   └── collect_barometric_pressure.py
│   ├── modeling/             # ML models and predictions
│   │   ├── fishing_forecast.py
│   │   ├── train_with_tidal.py
│   │   ├── feature_engineering.py
│   │   └── predict_fishing_conditions.py
│   ├── visualization/        # Plotting and analysis
│   │   ├── visualize_buoy_temps.py
│   │   └── visualize_model_behavior.py
│   └── utils/                # Shared utilities
│       └── sst_utils.py
├── scripts/                  # Pipeline orchestration
│   ├── run_pipeline.sh
│   └── run_enhanced_pipeline.py
├── data/                     # Data storage
│   ├── raw/noaa/            # Raw NOAA downloads
│   ├── processed/           # Cleaned datasets
│   └── features/            # ML feature matrices
├── models/                   # Trained models
├── figures/                  # Generated plots
├── docs/                     # Documentation
├── tests/                    # Unit tests
├── forecast.py              # Main entry point
├── docker-compose.yml
├── Dockerfile
├── Makefile
└── requirements.txt
```

## 🚀 Available Commands

### Make Commands (Docker)
```bash
make help              # Show all available commands
make build             # Build Docker images
make forecast          # Run fishing forecast
make train             # Train ML models
make collect           # Collect all NOAA data
make visualize         # Generate visualizations
make pipeline          # Run full pipeline
```

### Direct Python (Local venv)
```bash
python forecast.py                                      # Quick forecast
python src/modeling/fishing_forecast.py --forecast-days 7
python src/data_collection/pull_noaa_history.py         # Pull historical data
python src/visualization/visualize_buoy_temps.py        # Create plots
```

## 📊 Features

- **30+ years of NOAA data** (1993-present)
- **Random Forest ML model** with temperature forecasting
- **Species-specific scoring** (Rockfish, Halibut, Salmon, etc.)
- **Multi-source data integration** (SST, tides, weather, pressure)
- **Docker containerization** for reproducibility
- **Automated pipeline** for daily updates

## 🎯 Model Performance

- **Temperature Forecast R²:** ~0.85
- **Feature Importance:** Temperature lags, rolling means, cyclical seasonality
- **Prediction Window:** 7 days
- **Target Species:** 6 Monterey Bay species with optimal temp ranges

## 📚 Documentation

See `/docs` for:
- `COMPLETE_GUIDE.md` - Full usage guide
- `DOCKER_QUICKSTART.md` - Docker setup
- `SCIENTIFIC_REPORT.md` - Model methodology

## 🔧 Development

```bash
# Run tests
pytest tests/

# Check code style
flake8 src/

# Type checking
mypy src/
```

## 📝 License

MIT License - See LICENSE file for details.

## 🙏 Data Sources

- NOAA CO-OPS API (water temperature, tides, pressure)
- Historical SST archive (1993-present)
- Species profiles based on CA Department of Fish & Wildlife data
