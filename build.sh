#!/bin/bash
echo "🔧 Starting build process..."

# Print Python version
echo "Python version: $(python --version)"

# Print pip version
echo "Pip version: $(pip --version)"

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements_render.txt

# List installed packages
echo "📋 Installed packages:"
pip list

# Check if model file exists
if [ -f "best_model.pkl" ]; then
    echo "✅ Model file found"
else
    echo "❌ Model file not found!"
    exit 1
fi

echo "✅ Build completed successfully"
