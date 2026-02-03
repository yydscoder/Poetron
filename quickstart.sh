#!/bin/bash
# QUICK START - Run this to get Poetron up and running in 2 minutes

set -e  # Exit on error

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║             POETRON - QUICK START                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Navigate to project
cd Poetron || { echo "❌ Error: Cannot find Poetron directory"; exit 1; }

# Step 1: Check Python
echo "Step  Checking Python installation..."
python --version || { echo "❌ Python not found"; exit 1; }
echo "✅ Python found\n"

# Step 2: Create venv if needed
echo "Step 2 Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi
echo ""

# Step 3: Activate venv
echo "Step 3 Activating virtual environment..."
source venv/bin/activate || . venv/Scripts/activate
echo "✅ Virtual environment activated\n"

# Step 4: Install dependencies
echo "Step 4 Installing dependencies..."
pip install -r requirements.txt --quiet
echo "✅ Dependencies installed\n"

# Step 5: Run tests
echo "Step 5 Running test suite..."
echo ""
python ../test_project.py
echo ""

# Step 6: Show next steps
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║               ✅ SETUP COMPLETE!                           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "NEXT STEPS:"
echo ""
echo "1 Generate your first poem:"
echo "    python poetry_cli.py generate --style haiku --seed 'morning'"
echo ""
echo "2  List available styles:"
echo "    python poetry_cli.py list-styles"
echo ""
echo "3  Generate and export:"
echo "    python poetry_cli.py generate --style sonnet --export"
echo ""
echo "4  Use pre-trained Kaggle model:"
echo "    python setup_and_generate.py"
echo ""
echo "5 Train your own model locally:"
echo "    python download_data.py"
echo "    python src/train.py"
echo ""

