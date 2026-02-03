#!/bin/bash
# Quick startup script for Poetron

set -e  # Exit on error

echo "🎭 Welcome to Poetron - AI-Powered Poetry Generation System!"
echo "============================================================="

# Check if we're in the right directory
if [ ! -f "interactive_poet.py" ] || [ ! -d "src" ]; then
    echo "❌ Error: This script must be run from the Poetron directory"
    echo "Please navigate to the Poetron directory and run: bash start_poetron.sh"
    exit 1
fi

echo "🔍 Checking Python installation..."
python_version=$(python3 --version 2>&1 || python --version 2>&1)
if [ $? -ne 0 ]; then
    echo "❌ Python not found. Please install Python 3.7+"
    exit 1
fi
echo "✅ Python found: $python_version"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"

# Install dependencies if not already installed
if ! python -c "import transformers, peft, torch" &> /dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements_inference.txt
    echo "✅ Dependencies installed"
else
    echo "✅ Dependencies already installed"
fi

# Check if model exists
if [ ! -d "models/poetry_model" ]; then
    echo "❌ Model not found!"
    echo "Please download the model first by running:"
    echo "  bash download_kaggle_trained_model.sh"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo "✅ Model found!"

echo ""
echo "🎭 Starting Poetron - Interactive Poetry Generator!"
echo "==================================================="
echo ""

# Run the interactive poet
python interactive_poet.py

echo ""
echo "👋 Thank you for using Poetron! Goodbye! 🎭"