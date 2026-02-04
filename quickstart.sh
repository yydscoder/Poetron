#!/bin/bash
# QUICK START - Local Model Only Version (No Training, No API)

set -e  # Exit on error

# Load existing .env file if it exists
if [ -f ".env" ]; then
    echo " Loading existing API keys from .env file..."
    source .env
    echo "API keys loaded from .env file"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════......╝"
echo ""
echo "AI-Powered Poetry Generation System - Local Model Only"
echo ""
echo "Welcome to Poetron! This script sets up the poetry generation system"
echo "to run your trained model locally. This version focuses on inference only."
echo ""

# Prompt for API key (for poetry refinement)
echo "API Key Setup:"
echo "If you have an API key for enhanced poetry refinement, you can enter it now."
echo "Otherwise, press Enter to skip and use local-only features."
echo ""
read -s -p "Enter your POETRON_API_KEY (for poetry refinement, optional): " poetron_api_key
echo ""  # New line after hidden input

# Set environment variable if key was provided
if [ -n "$poetron_api_key" ]; then
    export POETRON_API_KEY="$poetron_api_key"
    echo " POETRON_API_KEY set for this session"
fi

# Ask if user wants to save API key to a .env file for persistent use
if [ -n "$poetron_api_key" ]; then
    echo ""
    read -p "Do you want to save this API key to a .env file for future use? (y/n): " save_keys
    if [[ $save_keys =~ ^[Yy]$ ]]; then
        echo "# Poetron API Keys - $(date)" > .env
        echo "POETRON_API_KEY=$poetron_api_key" >> .env
        echo " API key saved to .env file"
        echo " Tip: You can load this key in future sessions with: source .env"
    fi
fi

echo ""
# Check available disk space (require at least 500MB free for heavy install)
available_space=$(df . | awk 'NR==2 {print $4}' | sed 's/K$//')
available_mb=$((available_space / 1024))

echo " Disk Space Analysis:"
echo "   Available space: ~${available_mb}MB"
echo ""

if [ "$available_mb" -lt 500 ]; then
    echo "WARNING: Insufficient disk space detected (< 500MB available)"
    echo "   This may cause installation failures with heavy dependencies."
    echo ""
    echo "This script will install the local model version (requires ~1.5GB+ space)."
    echo "Press Enter to continue or Ctrl+C to exit if you don't have enough space."
    read -p ""
else
    echo "This script will install the local model version (requires ~1.5GB+ space)."
    echo "Press Enter to continue..."
    read -p ""
fi

# Navigate to project
echo ""
echo " Verifying project directory..."
cd Poetron || { echo " Error: Cannot find Poetron directory"; exit 1; }
echo " Project directory verified"
echo ""

# Step 1: Check Python
echo "Step 1  Checking Python installation..."
python_version=$(python --version 2>&1)
if [ $? -eq 0 ]; then
    echo "Python found: $python_version"
else
    echo " Python not found. Please install Python 3.7+"
    exit 1
fi
echo ""

# Step 2: Create venv if needed
echo " Step 2 Setting up virtual environment..."
if [ ! -d "venv" ]; then
    echo "   Creating new virtual environment..."
    python -m venv venv
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi
echo ""

# Step 3: Activate venv
echo "🔌 Step 3️⃣  Activating virtual environment..."
source venv/bin/activate || . venv/Scripts/activate
echo " Virtual environment activated"
echo ""

# Step 4: Install inference-only dependencies (CPU version)
echo " Step 4  Installing inference-only dependencies..."
echo "   Installing PyTorch CPU-only version (faster installation)..."
pip install torch>=2.1.0 --index-url https://download.pytorch.org/whl/cpu --verbose
echo "    PyTorch CPU-only installed"

echo "   Installing Transformers (model handling)..."
pip install transformers>=4.44.0 --verbose
echo "    Transformers installed"

echo "   Installing PEFT (Parameter Efficient Fine-Tuning for using your trained model)..."
pip install peft>=0.11.0 --verbose
echo "    PEFT installed"

echo "   Installing Tokenizers (text processing)..."
pip install tokenizers>=0.19.0 --verbose
echo "    Tokenizers installed"

echo "   Installing HuggingFace Hub (for model downloads)..."
pip install huggingface_hub>=0.25.1 --verbose
echo "    HuggingFace Hub installed"

echo "   Installing Click (for CLI interface)..."
pip install click>=8.0.0 --verbose
echo "   Click installed"

echo "   Installing Requests (for HTTP requests)..."
pip install requests>=2.25.0 --verbose
echo "    Requests installed"

echo "   Installing KaggleHub (for Kaggle model downloads)..."
pip install kagglehub>=0.2.0 --verbose
echo "    KaggleHub installed"

echo " All inference-only dependencies installed (~1.0-1.5GB total)"
echo ""

# Step 5: Download trained Kaggle model
echo " Step 5  Downloading pre-trained Kaggle model..."
echo "   This model was trained specifically for poetry generation."
mkdir -p models/kaggle_trained_model

if [ ! -f ~/Downloads/flavourtownpoetrongeneratormodel.zip ]; then
    echo "    Fetching model from Kaggle (this may take 1-2 minutes)..."
    echo "   Model size: ~1.1GB"
    curl -L -o ~/Downloads/flavourtownpoetrongeneratormodel.zip \
      https://www.kaggle.com/api/v1/datasets/download/xongkoro/flavourtownpoetrongeneratormodel
else
    echo "    Using cached model file from ~/Downloads/"
fi

echo "    Extracting model files..."
unzip -q ~/Downloads/flavourtownpoetrongeneratormodel.zip -d models/kaggle_trained_model
echo "Model downloaded and extracted successfully"
echo ""

# Create symbolic link for default model path (CORRECTED PATH)
echo "Creating model path link..."
mkdir -p ./models
rm -rf ./models/poetry_model 2>/dev/null || true
# Use relative path from models/ directory to the actual model location
ln -s "kaggle_trained_model/kaggle/working/poetry_model/final_poetry_lora/" ./models/poetry_model
echo " Model path linked successfully"
echo ""

# Verify the link is working
echo " Verifying model path..."
if [ -f "./models/poetry_model/adapter_config.json" ]; then
    echo "Model verification successful - files found"
else
    echo "Model verification failed - files not accessible"
    exit 1
fi
echo ""

# Test local generation
echo "🧪 Step 6️⃣  Testing local poem generation..."
echo "   Generating a test haiku to verify everything works..."
python poetry_cli.py generate --style haiku --seed "test"
echo ""

# Show next steps
echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════......╝"
echo ""
echo "Your Poet is awake and ready!"
echo ""

echo "LOCAL MODEL MODE - NEXT COMMANDS:"
echo "1  Generate a haiku:"
echo "    python poetry_cli.py generate --style haiku --seed 'moonlight'"
echo ""
echo "2  Generate a sonnet:"
echo "    python poetry_cli.py generate --style sonnet --seed 'love'"
echo ""
echo "3  Generate free verse:"
echo "    python poetry_cli.py generate --style freeverse --seed 'ocean'"
echo ""
echo "4  Export to file:"
echo "    python poetry_cli.py generate --style haiku --export"
echo ""
echo "5 List available styles:"
echo "    python poetry_cli.py list-styles"
echo ""
echo "6 Use API refinement (if you provided API key):"
echo "    The API key will be used automatically when refining poems"
echo ""
echo "🚀 PRO TIP: Your trained model is now ready to generate unique poems!"
echo "   The model was trained on poetry data and captures distinctive style."
echo ""
echo "NOTE: This installation is optimized for inference only."
echo "   Training capabilities have been removed to save space."
echo ""
echo "API KEY: If you provided an API key during setup, it is available"
echo "   for use in this session. If you saved it to a .env file, you can load"
echo "   it in future sessions with: source .env"