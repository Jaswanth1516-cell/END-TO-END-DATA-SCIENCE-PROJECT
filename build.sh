#!/bin/bash

# Build script for local testing
set -e

echo "🔨 Building End-to-End Data Science Project..."

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Train model
echo "🤖 Training model..."
python model_trainer.py

# Run tests
echo "🧪 Running tests..."
python test_api.py

echo "✅ Build complete!"
echo ""
echo "Next steps:"
echo "  1. Run Flask: python flask_app.py"
echo "  2. Or run FastAPI: uvicorn fastapi_app:app --reload"
echo "  3. Visit: http://localhost:5000 or http://localhost:8000"
