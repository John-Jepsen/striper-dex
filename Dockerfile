FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./
COPY *.sh ./
COPY data/ ./data/
COPY models/ ./models/
COPY figures/ ./figures/
COPY archive/ ./archive/

# Make shell scripts executable
RUN chmod +x *.sh 2>/dev/null || true
RUN chmod +x *.py 2>/dev/null || true

# Create necessary directories
RUN mkdir -p data/raw/noaa data/processed data/features models figures

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command
CMD ["python", "train_with_tidal.py"]
