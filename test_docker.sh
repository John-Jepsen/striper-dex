#!/bin/bash
set -e

echo "🐳 Testing Docker Setup for Bay Water Temps"
echo "============================================"
echo ""

# Check Docker is installed
echo "✓ Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi
echo "  Docker version: $(docker --version)"

# Check Docker Compose is installed
echo ""
echo "✓ Checking Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi
echo "  Docker Compose version: $(docker-compose --version)"

# Validate docker-compose.yml
echo ""
echo "✓ Validating docker-compose.yml..."
if docker-compose config --quiet; then
    echo "  ✓ Configuration is valid"
else
    echo "  ❌ Configuration has errors"
    exit 1
fi

# Check Makefile
echo ""
echo "✓ Checking Makefile..."
if [ -f "Makefile" ]; then
    echo "  ✓ Makefile exists"
else
    echo "  ❌ Makefile not found"
    exit 1
fi

# Check Dockerfile
echo ""
echo "✓ Checking Dockerfile..."
if [ -f "Dockerfile" ]; then
    echo "  ✓ Dockerfile exists"
else
    echo "  ❌ Dockerfile not found"
    exit 1
fi

# List available services
echo ""
echo "✓ Available Docker services:"
docker-compose config --services | sed 's/^/  - /'

echo ""
echo "============================================"
echo "✅ Docker setup is ready!"
echo ""
echo "Next steps:"
echo "  1. make build    # Build the Docker image"
echo "  2. make train    # Train the model"
echo "  3. make help     # See all commands"
echo ""
