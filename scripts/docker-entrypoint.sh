#!/bin/bash
set -e

# Create necessary directories
mkdir -p /app/data/raw/noaa
mkdir -p /app/data/processed
mkdir -p /app/data/features
mkdir -p /app/models
mkdir -p /app/figures

# Execute the command
exec "$@"
