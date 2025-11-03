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
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY models/ ./models/
COPY figures/ ./figures/
COPY requirements.txt ./

# Make shell scripts executable
RUN chmod +x scripts/*.sh 2>/dev/null || true

# Create necessary directories
RUN mkdir -p data/raw/noaa data/processed data/features models figures

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command
CMD ["python", "src/modeling/train_with_tidal.py"]
