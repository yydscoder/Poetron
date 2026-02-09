#!/usr/bin/env python3
"""
Setup and Run Script for Poetron - Poetry Generation System
This script replicates the functionality of quickstart.sh for Windows users.
Installs dependencies, downloads the model, and runs the interactive poet.
"""

import subprocess
import sys
import os
from pathlib import Path
import zipfile
import requests
from urllib.parse import urlparse
import shutil


def load_env_file():
    """Load existing .env file if it exists"""
    if Path(".env").exists():
        print("[INFO] Loading existing API keys from .env file...")
        with open(".env", "r") as env_file:
            for line in env_file:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
        print("[SUCCESS] API keys loaded from .env file")


def prompt_for_api_key():
    """Prompt for API key and optionally save to .env file"""
    print("\nAPI Key Setup:")
    print("If you have an API key for enhanced poetry refinement, you can enter it now.")
    print("Otherwise, press Enter to skip and use local-only features.")

    api_key = input("\nEnter your POETRON_API_KEY (for poetry refinement, optional): ")

    if api_key:
        os.environ["POETRON_API_KEY"] = api_key
        print(" [SUCCESS] POETRON_API_KEY set for this session")

        # Ask if user wants to save API key to a .env file
        save_keys = input("\nDo you want to save this API key to a .env file for future use? (y/n): ")
        if save_keys.lower() in ['y', 'yes']:
            with open(".env", "a") as env_file:
                env_file.write(f"\n# Poetron API Keys - {os.popen('date').read().strip()}\n")
                env_file.write(f"POETRON_API_KEY={api_key}\n")
            print("[SUCCESS] API key saved to .env file")


def check_disk_space():
    """Check available disk space"""
    import shutil
    total, used, free = shutil.disk_usage(".")
    free_mb = free // (1024 * 1024)

    print(f"\n Disk Space Analysis:")
    print(f"   Available space: ~{free_mb}MB")

    if free_mb < 500:
        print("\n[WARNING] Insufficient disk space detected (< 500MB available)")
        print("   This may cause installation failures with heavy dependencies.")
        print("\nThis script will install the local model version (requires ~1.5GB+ space).")
        input("Press Enter to continue or Ctrl+C to exit if you don't have enough space.")
    else:
        print("\nThis script will install the local model version (requires ~1.5GB+ space).")
        input("Press Enter to continue...")


def install_dependencies():
    """Install required dependencies for inference only (CPU version)."""
    print("\n[INFO] Installing inference-only dependencies...")

    # Install PyTorch CPU-only version first (faster installation)
    print("   Installing PyTorch CPU-only version (faster installation)...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "torch>=2.1.0",
            "--index-url", "https://download.pytorch.org/whl/cpu", "--verbose"
        ])
        print("    [SUCCESS] PyTorch CPU-only installed")
    except subprocess.CalledProcessError:
        print("[ERROR] Failed to install PyTorch")
        return False

    # Install other dependencies
    packages = [
        ("transformers>=4.44.0", "Transformers (model handling)"),
        ("peft>=0.11.0", "PEFT (Parameter Efficient Fine-Tuning for using your trained model)"),
        ("tokenizers>=0.19.0", "Tokenizers (text processing)"),
        ("huggingface_hub>=0.25.1", "HuggingFace Hub (for model downloads)"),
        ("click>=8.0.0", "Click (for CLI interface)"),
        ("requests>=2.25.0", "Requests (for HTTP requests)"),
        ("kagglehub>=0.2.0", "KaggleHub (for Kaggle model downloads)")
    ]

    for package, description in packages:
        print(f"   Installing {description}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--verbose"])
            print(f"   [SUCCESS] {description.split(' ')[0]} installed")
        except subprocess.CalledProcessError:
            print(f"[ERROR] Failed to install {package}")
            return False

    print("\n[SUCCESS] All inference-only dependencies installed (~1.0-1.5GB total)")
    return True


def download_kaggle_model():
    """Download the trained Kaggle model"""
    print("\n[INFO] Downloading pre-trained Kaggle model...")
    print("   This model was trained specifically for poetry generation.")

    # Create models directory
    models_dir = Path("./models/kaggle_trained_model")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Define download URL and path
    download_url = "https://www.kaggle.com/api/v1/datasets/download/xongkoro/flavourtownpoetrongeneratormodel"
    downloads_path = Path.home() / "Downloads"
    model_zip_path = downloads_path / "flavourtownpoetrongeneratormodel.zip"

    # Download the model if not already present
    if not model_zip_path.exists():
        print("    Fetching model from Kaggle (this may take 1-2 minutes)...")
        print("   Model size: ~1.1GB")

        try:
            response = requests.get(download_url)
            response.raise_for_status()

            with open(model_zip_path, 'wb') as f:
                f.write(response.content)

            print("[SUCCESS] Model downloaded successfully")
        except Exception as e:
            print(f"[ERROR] Error downloading model: {e}")
            return False
    else:
        print("    Using cached model file from ~/Downloads/")

    # Extract the model
    print("    Extracting model files...")
    try:
        with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
            zip_ref.extractall(models_dir)
        print("[SUCCESS] Model downloaded and extracted successfully")
    except Exception as e:
        print(f"[ERROR] Error extracting model: {e}")
        return False

    # Create model path link/copy
    print("\nCreating model path link...")
    final_model_path = models_dir / "kaggle" / "working" / "poetry_model" / "final_poetry_lora"

    poetry_models_dir = Path("./models")
    poetry_models_dir.mkdir(exist_ok=True)

    target_model_path = poetry_models_dir / "poetry_model"

    # On Windows, we'll copy instead of creating a symlink
    if target_model_path.exists():
        shutil.rmtree(target_model_path)

    if final_model_path.exists():
        shutil.copytree(final_model_path, target_model_path)
        print("[SUCCESS] Model path copied successfully")
    else:
        print(f"[ERROR] Expected model path does not exist: {final_model_path}")
        return False

    # Verify the model is accessible
    print("\nVerifying model path...")
    if (target_model_path / "adapter_config.json").exists():
        print("[SUCCESS] Model verification successful - files found")
    else:
        print("[ERROR] Model verification failed - files not accessible")
        return False

    return True


def test_local_generation():
    """Test local poem generation"""
    print("\n[INFO] Testing local poem generation...")
    print("   Generating a test haiku to verify everything works...")

    try:
        # Try to run the CLI to generate a test haiku
        result = subprocess.run([
            sys.executable, "poetry_cli.py", "generate", "--style", "haiku", "--seed", "test"
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("[SUCCESS] Test generation successful!")
            print(result.stdout)
        else:
            print("[WARNING] Test generation had some issues, but continuing...")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
    except Exception as e:
        print(f"[WARNING] Could not run test generation: {e}")


def run_interactive_poet():
    """Run the interactive poet."""
    print("\n[INFO] Starting Poetron - Interactive Poetry Generator!")
    print("=" * 60)

    # Import and run the interactive poet
    try:
        from interactive_poet import main
        main()
    except ImportError as e:
        print(f"[ERROR] Could not import interactive_poet: {e}")
        print("Please make sure interactive_poet.py exists in the current directory.")
    except Exception as e:
        print(f"[ERROR] Error running interactive poet: {e}")


def show_next_steps():
    """Show next steps to the user"""
    print("\n" + "="*120)
    print("[SUCCESS] Your Poet is awake and ready!")
    print("="*120)

    print("\nLOCAL MODEL MODE - NEXT COMMANDS:")
    print("1. Generate a haiku:")
    print("    python poetry_cli.py generate --style haiku --seed 'moonlight'")
    print("")
    print("2. Generate a sonnet:")
    print("    python poetry_cli.py generate --style sonnet --seed 'love'")
    print("")
    print("3. Generate free verse:")
    print("    python poetry_cli.py generate --style freeverse --seed 'ocean'")
    print("")
    print("4. Generate with export to file:")
    print("    python poetry_cli.py generate --style haiku --export")
    print("")
    print("5. List available styles:")
    print("    python poetry_cli.py list-styles")
    print("")
    print("6. Use API refinement (if you provided API key):")
    print("    The API key will be used automatically when refining poems")
    print("")
    print("PRO TIP: Your trained model is now ready to generate unique poems!")
    print("   The model was trained on poetry data and captures distinctive style.")
    print("")
    print("NOTE: This installation is optimized for inference only.")
    print("   Training capabilities have been removed to save space.")
    print("")
    if "POETRON_API_KEY" in os.environ:
        print("API KEY: The API key you provided is available for use in this session.")


def main():
    """Main setup and run function that replicates quickstart.sh functionality."""
    print("\n" + "="*120)
    print("AI-Powered Poetry Generation System - Local Model Only")
    print("="*120)
    print("\nWelcome to Poetron! This script sets up the poetry generation system")
    print("to run your trained model locally. This version focuses on inference only.")
    print()

    # Load existing .env file if it exists
    load_env_file()

    # Prompt for API key
    prompt_for_api_key()

    # Check disk space
    check_disk_space()

    # Verify we're in the right directory
    if not Path("interactive_poet.py").exists():
        print("[ERROR] Cannot find interactive_poet.py in current directory")
        print("Make sure you're running this script from the Poetron directory")
        return

    # Check Python
    print("\n[INFO] Step 1: Checking Python installation...")
    python_version = subprocess.check_output([sys.executable, "--version"]).decode().strip()
    print(f"[SUCCESS] Python found: {python_version}")

    # Install dependencies
    print("\n[INFO] Step 4: Installing inference-only dependencies...")
    if not install_dependencies():
        print("[ERROR] Failed to install dependencies. Exiting.")
        return

    # Download trained model
    print("\n[INFO] Step 5: Downloading pre-trained Kaggle model...")
    if not download_kaggle_model():
        print("[ERROR] Failed to setup model. Exiting.")
        return

    # Test local generation
    print("\n[INFO] Step 6: Testing local poem generation...")
    test_local_generation()

    # Show next steps
    show_next_steps()

    print("\n[INFO] Starting interactive poet mode...")
    # Run the interactive poet
    run_interactive_poet()


if __name__ == "__main__":
    main()