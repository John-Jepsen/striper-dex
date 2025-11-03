#!/usr/bin/env python3
"""
Main entry point for Monterey Bay Fishing Forecast.

This script provides a simple CLI to run the fishing forecast with the latest data.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.modeling.fishing_forecast import main

if __name__ == "__main__":
    sys.exit(main() or 0)
