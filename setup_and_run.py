"""
Poetron - Automated Setup and Launch Script
This script handles complete setup and runs the interactive haiku generator
"""
import subprocess
import sys
import os
from pathlib import Path

def print_header(message):
    print("\n" + "="*70)
    print(f"  {message}")
    print("="*70)

def check_python_version():
    """Ensure Python version is 3.8+"""
    print("Checking Python version...")
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_dependencies():
    """Install required Python packages"""
    print("\nInstalling dependencies (this may take a few minutes)...")
    
    packages = [
        "torch>=2.1.0",
        "transformers>=4.44.0",
        "peft>=0.11.0",
        "tokenizers>=0.19.0",
        "huggingface_hub>=0.25.1",
        "click>=8.0.0",
        "requests>=2.25.0"
    ]
    
    try:
        # Upgrade pip first
        print("Upgrading pip...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Install packages
        print("Installing required packages...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install"
        ] + packages, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install dependencies: {e}")
        print("\nPlease install manually:")
        print("pip install torch transformers peft tokenizers huggingface_hub click requests")
        return False

def check_models_exist():
    """Check if any model files exist"""
    model_paths = [
        Path("models/poetry_model"),
        Path("models/kaggle_trained_model")
    ]
    
    for path in model_paths:
        if path.exists() and any(path.iterdir()):
            return True
    return False

def run_interactive_haiku():
    """Launch the interactive haiku generator"""
    print("\nLaunching interactive haiku generator...")
    print("Note: First-time AI model loading may take a moment")
    print("      (rule-based generator is instant)\n")
    
    try:
        # Run the interactive haiku generator
        subprocess.call([sys.executable, "interactive_haiku.py"])
    except KeyboardInterrupt:
        print("\n\nGenerator closed.")
    except Exception as e:
        print(f"ERROR: Failed to run generator: {e}")
        sys.exit(1)

def main():
    """Main setup and run workflow"""
    print_header("POETRON - AI Haiku Generator Setup")
    
    # Step 1: Check Python version
    check_python_version()
    
    # Step 2: Install dependencies
    print_header("Installing Dependencies")
    if not install_dependencies():
        sys.exit(1)
    
    # Step 3: Check for models
    print_header("Checking Models")
    if check_models_exist():
        print("Model files found - AI generation available")
    else:
        print("No model files found - will use rule-based generator")
        print("(Rule-based generator produces quality haikus)")
    
    # Step 4: Launch interactive generator
    print_header("Setup Complete - Starting Generator")
    run_interactive_haiku()

if __name__ == "__main__":
    main()
