#!/usr/bin/env python3
"""
Uninstallation Script for Poetron - Poetry Generation System
This script removes all installed components and dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path
import shutil


def uninstall_dependencies():
    """Uninstall all dependencies installed by the setup."""
    print("[INFO] Uninstalling Poetron dependencies...")

    # List of packages to uninstall
    packages = [
        "torch",
        "transformers", 
        "peft",
        "tokenizers",
        "huggingface_hub",
        "click",
        "requests",
        "kagglehub"
    ]

    for package in packages:
        print(f"   Uninstalling {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", package])
            print(f"   [SUCCESS] {package} uninstalled")
        except subprocess.CalledProcessError:
            print(f"   [WARNING] Could not uninstall {package} (may not be installed)")

    print("[SUCCESS] Dependencies uninstalled!")


def remove_model_files():
    """Remove model files and directories."""
    print("\n[INFO] Removing model files...")
    
    model_dirs = [
        Path("./models"),
        Path("./venv")
    ]
    
    for model_dir in model_dirs:
        if model_dir.exists():
            try:
                shutil.rmtree(model_dir)
                print(f"   [SUCCESS] Removed {model_dir}")
            except Exception as e:
                print(f"   [ERROR] Could not remove {model_dir}: {e}")
        else:
            print(f"   [INFO] {model_dir} does not exist, skipping...")
    
    print("[SUCCESS] Model files removed!")


def remove_env_file():
    """Remove .env file if it exists."""
    print("\n[INFO] Checking for .env file...")
    
    env_file = Path(".env")
    if env_file.exists():
        try:
            env_file.unlink()
            print("   [SUCCESS] Removed .env file")
        except Exception as e:
            print(f"   [ERROR] Could not remove .env file: {e}")
    else:
        print("   [INFO] .env file does not exist, skipping...")
    
    print("[SUCCESS] Environment file handling complete!")


def main():
    """Main uninstallation function."""
    print("\n" + "="*60)
    print("Poetron - Uninstallation Script")
    print("="*60)
    print("\nThis script will remove:")
    print("1. All installed Python dependencies")
    print("2. Model files and directories")
    print("3. Virtual environment (if created)")
    print("4. .env file (if exists)")
    print("\nWARNING: This action cannot be undone!")
    print("="*60)
    
    confirm = input("\nDo you want to proceed with uninstallation? (yes/no): ")
    
    if confirm.lower() not in ['yes', 'y']:
        print("[INFO] Uninstallation cancelled by user.")
        return
    
    # Uninstall dependencies
    uninstall_dependencies()
    
    # Remove model files
    remove_model_files()
    
    # Remove env file
    remove_env_file()
    
    print("\n" + "="*60)
    print("[SUCCESS] Poetron has been uninstalled successfully!")
    print("All components have been removed from your system.")
    print("="*60)


if __name__ == "__main__":
    main()