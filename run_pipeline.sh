#!/bin/bash
# Complete pipeline to run the entire fishing prediction project

set -e  # Exit on error

echo "=========================================="
echo "Monterey Bay Fishing Prediction Pipeline"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3.8+."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "=========================================="
echo "Step 1: Collect Historical Data"
echo "=========================================="
echo ""

# Collect water temperature data
if [ ! -f "data/processed/9413450_morning_daily.csv" ]; then
    echo "Pulling water temperature history (this may take a few minutes)..."
    python pull_noaa_history.py \
        --station 9413450 \
        --start 2020-01-01 \
        --chunk-days 365 \
        --pause 0.5
else
    echo "✓ Temperature data already collected"
fi

# Collect barometric pressure data
if [ ! -f "data/processed/9413450_barometric_pressure.csv" ]; then
    echo "Pulling barometric pressure history..."
    python collect_barometric_pressure.py \
        --station 9413450 \
        --start 2020-01-01 \
        --chunk-days 30 \
        --pause 0.5
else
    echo "✓ Pressure data already collected"
fi

echo ""
echo "=========================================="
echo "Step 2: Feature Engineering"
echo "=========================================="
echo ""

if [ ! -f "data/features/fishing_features.csv" ]; then
    echo "Creating ML features from raw data..."
    python feature_engineering.py \
        --temp-file data/processed/9413450_morning_daily.csv \
        --pressure-file data/processed/9413450_barometric_pressure.csv \
        --output data/features/fishing_features.csv
else
    echo "✓ Features already engineered"
    echo "  To regenerate, delete data/features/fishing_features.csv"
fi

echo ""
echo "=========================================="
echo "Step 3: Train ML Models"
echo "=========================================="
echo ""

echo "Training models (this may take 1-2 minutes)..."
python train_fishing_model.py \
    --features data/features/fishing_features.csv \
    --model all \
    --output-dir models \
    --test-size 0.2

echo ""
echo "=========================================="
echo "Step 4: Hypothesis Testing"
echo "=========================================="
echo ""

echo "Running statistical tests..."
python hypothesis_testing.py \
    --features data/features/fishing_features.csv \
    --output-dir reports

echo ""
echo "=========================================="
echo "Step 5: Generate Visualizations"
echo "=========================================="
echo ""

# Visualize temperature trends if weekly data exists
if [ -f "monterey_morning_temps_weekly_last150days.csv" ]; then
    echo "Creating temperature visualizations..."
    python visualize_buoy_temps.py \
        --input monterey_morning_temps_weekly_last150days.csv \
        --outdir figures \
        --dpi 140
else
    echo "Note: Run buoy-temps.py separately to generate weekly temperature plots"
fi

echo ""
echo "=========================================="
echo "Step 6: Make Predictions"
echo "=========================================="
echo ""

echo "Generating 7-day fishing forecast..."
python predict_fishing_conditions.py \
    --model-dir models \
    --forecast 7

echo ""
echo "=========================================="
echo "✅ PIPELINE COMPLETE!"
echo "=========================================="
echo ""
echo "Generated outputs:"
echo "  📊 Models:        models/"
echo "  📈 Figures:       figures/"
echo "  📋 Reports:       reports/"
echo "  📦 Data:          data/"
echo ""
echo "Next steps:"
echo "  1. Review SCIENTIFIC_REPORT.md for full analysis"
echo "  2. Check figures/ for visualizations"
echo "  3. Use predict_fishing_conditions.py for daily forecasts"
echo ""
echo "To run individual steps:"
echo "  python collect_barometric_pressure.py --help"
echo "  python feature_engineering.py --help"
echo "  python train_fishing_model.py --help"
echo "  python hypothesis_testing.py --help"
echo "  python predict_fishing_conditions.py --help"
echo ""
