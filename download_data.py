#!/usr/bin/env python3
"""
This is a script to download the poetry dataset from Kaggle, we are using https://www.kaggle.com/datasets/tgdivy/poetry-foundation-poems
"""

import os
import sys
from pathlib import Path

def download_dataset():
    """Downloading the poetry dataset from Kaggle"""
    try:
        import kagglehub
        print("Downloading poetry dataset, BE PATIENT")
        
        # Download latest version
        path = kagglehub.dataset_download("tgdivy/poetry-foundation-poems")
        
        print(f"Path to dataset files: {path}")
        
        # List files in the downloaded directory
        print("\nFiles in dataset:")
        for file_path in Path(path).rglob("*"):
            print(f"  {file_path}")
            
        # Copy files to our data directory
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        for file_path in Path(path).rglob("*"):
            if file_path.is_file():
                dest_path = data_dir / file_path.name
                print(f"Copying {file_path} to {dest_path}")
                dest_path.write_bytes(file_path.read_bytes())
        
        print(f"\nDataset copied to {data_dir.absolute()}")
        return str(data_dir.absolute())
        
    except ImportError:
        print("kagglehub not installed. Please install it using: pip install kagglehub")
        return None
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

if __name__ == "__main__":
    download_dataset()