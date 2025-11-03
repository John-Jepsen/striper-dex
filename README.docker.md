# Docker Setup for Monterey Bay Fishing Prediction

## Quick Start

```bash
# Build the Docker image
make build

# Train the model
make train

# That's it! Model will be saved to ./models/
```

---

## Prerequisites

- Docker installed
- Docker Compose installed
- 2GB free disk space

---

## Usage

### Training

```bash
# Train with tidal data (R²=0.72)
make train

# Train production model
make train-prod
```

### Data Collection

```bash
# Collect all data sources
make collect

# Or collect individually
make collect-noaa      # NOAA historical data
make collect-buoy      # Buoy temperatures
make collect-weather   # Weather data
make collect-tidal     # Tidal patterns
make collect-pressure  # Barometric pressure
```

### Pipeline

```bash
# Run full pipeline
make pipeline

# Generate features only
make features
```

### Analysis

```bash
# Compare models
make compare

# Visualize data
make visualize
make visualize-model
```

### Predictions

```bash
# Generate forecast
make forecast

# Predict conditions
make predict
```

---

## Commands Reference

| Command | Description |
|---------|-------------|
| `make build` | Build Docker images |
| `make train` | Train model with tidal data |
| `make collect` | Collect all data |
| `make pipeline` | Run full pipeline |
| `make forecast` | Generate fishing forecast |
| `make shell` | Open bash shell in container |
| `make logs` | View logs |
| `make clean` | Remove containers/volumes |

---

## Directory Structure

```
.
├── data/          # Mounted volume for data
│   ├── raw/       # Raw NOAA data
│   ├── processed/ # Processed data
│   └── features/  # ML features
├── models/        # Trained models (saved here)
└── figures/       # Generated visualizations
```

**Note**: Data, models, and figures are persisted on your host machine via Docker volumes.

---

## Advanced Usage

### Run specific service

```bash
docker-compose run --rm <service-name>
```

Available services:
- `train` - Train with tidal data
- `collect-data` - NOAA data collection
- `features` - Feature engineering
- `forecast` - Fishing forecast
- `predict` - Condition predictions

### View logs

```bash
make logs SERVICE=train
```

### Open shell in container

```bash
make shell
```

### Use Docker Compose directly

```bash
# Train
docker-compose run --rm train

# Collect data
docker-compose --profile data-collection run --rm collect-tidal

# Pipeline
docker-compose --profile pipeline run --rm pipeline
```

---

## Profiles

Docker Compose uses profiles to organize services:

- **default**: Training services
- **data-collection**: Data collection services
- **pipeline**: Pipeline and feature engineering
- **analysis**: Visualization and model comparison
- **prediction**: Forecasting and predictions
- **production**: Production model training

Activate profiles with `--profile`:
```bash
docker-compose --profile analysis run --rm visualize
```

Or use the Makefile shortcuts.

---

## Troubleshooting

### Build fails

```bash
# Clean and rebuild
make clean
make build
```

### Permission issues with volumes

```bash
# Fix permissions (Linux/Mac)
sudo chown -R $USER:$USER data/ models/ figures/
```

### Out of disk space

```bash
# Clean up Docker
docker system prune -a
```

### Container won't start

```bash
# Check logs
make logs

# Or specific service
docker-compose logs train
```

---

## Environment Variables

Create a `.env` file for custom configuration:

```bash
# .env
PYTHONUNBUFFERED=1
MODEL_PATH=/app/models
DATA_PATH=/app/data
```

---

## Development

### Modify code without rebuilding

Code changes require rebuild:
```bash
make build
make train
```

### Live development

Mount source code for live changes:
```bash
docker-compose run --rm -v $(pwd):/app train python your_script.py
```

---

## Production Deployment

```bash
# Build for production
docker-compose build --no-cache

# Run production training
make train-prod

# Deploy model
# Models are in ./models/ directory
```

---

## Notes

- **Data persistence**: All data/models saved to host via volumes
- **Automatic cleanup**: Containers removed after completion (`--rm`)
- **Resource usage**: ~2GB RAM for training
- **Training time**: 2-5 minutes depending on hardware

---

## Getting Help

```bash
make help
```

Lists all available commands and descriptions.
