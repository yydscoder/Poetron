#!/usr/bin/env python3
"""
Setup and Run Script for Poetron - Poetry Generation System
This script installs dependencies and runs the interactive poet.
"""

import subprocess
import sys
import os
from pathlib import Path


def install_dependencies():
    """Install required dependencies for inference only."""
    print("📦 Installing required dependencies...")
    
    # Install inference-only dependencies
    packages = [
        "torch>=2.1.0",
        "transformers>=4.44.0", 
        "peft>=0.11.0",
        "tokenizers>=0.19.0",
        "huggingface_hub>=0.25.1",
        "click>=8.0.0",
        "requests>=2.25.0",
        "kagglehub>=0.2.0"
    ]
    
    for package in packages:
        print(f"   Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
            return False
    
    print("✅ Dependencies installed successfully!")
    return True


def check_model_exists():
    """Check if the trained model exists."""
    model_path = Path("./models/poetry_model")
    return model_path.exists()


def download_model_if_needed():
    """Download the model if it doesn't exist."""
    if check_model_exists():
        print("✅ Trained model already exists!")
        return True
    
    print("📦 Downloading pre-trained model...")
    
    # Create models directory
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    
    # Download the model
    try:
        # Check if we have the Kaggle trained model structure
        kaggle_model_path = Path("./models/kaggle_trained_model/kaggle/working/poetry_model/final_poetry_lora/")
        if kaggle_model_path.exists():
            # Create symbolic link
            poetry_model_link = Path("./models/poetry_model")
            if not poetry_model_link.exists():
                poetry_model_link.symlink_to(kaggle_model_path, target_is_directory=True)
                print("✅ Created model link!")
        else:
            print("❌ Pre-trained model not found. Please run: bash download_kaggle_trained_model.sh")
            return False
        
        print("✅ Model setup complete!")
        return True
    except Exception as e:
        print(f"❌ Error setting up model: {e}")
        return False


def run_interactive_poet():
    """Run the interactive poet."""
    print("\n🎭 Starting Poetron - Interactive Poetry Generator!")
    print("=" * 60)
    
    # Import and run the interactive poet
    try:
        from interactive_poet import main
        main()
    except ImportError as e:
        print(f"❌ Could not import interactive_poet: {e}")
        print("Please make sure interactive_poet.py exists in the current directory.")
    except Exception as e:
        print(f"❌ Error running interactive poet: {e}")


def main():
    """Main setup and run function."""
    print("🎭 Welcome to Poetron - AI-Powered Poetry Generation System!")
    print("=" * 60)
    
    # Check if we're in the right directory
    required_files = ["interactive_poet.py", "src/poetry_generator.py"]
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Please make sure you're running this script from the Poetron directory.")
        return
    
    print("🔍 Checking system requirements...")
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies. Exiting.")
        return
    
    # Check and setup model
    if not download_model_if_needed():
        print("❌ Failed to setup model. Please download the model first.")
        print("Run: bash download_kaggle_trained_model.sh")
        return
    
    print("\n✅ Setup complete! Starting interactive poet...")
    
    # Run the interactive poet
    run_interactive_poet()


if __name__ == "__main__":
    main()