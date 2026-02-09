#!/usr/bin/env python3
"""
Quick start script: Download model, load it, and generate poems locally
"""
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a shell command and report status"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"{description} failed")
        return False
    return True

def main():
    # Get repo root
    repo_root = Path(__file__).parent.parent
    
    # Step 1: Download model
    print("\n" + "="*60)
    print("FLAVOURTOWN POETRY GENERATOR - LOCAL SETUP")
    print("="*60)
    
    download_script = repo_root / "download_kaggle_trained_model.sh"
    if not run_command(f"bash {download_script}", "Downloading Kaggle model"):
        return False
    
    # Step 2: Load and test
    print(f"\n{'='*60}")
    print("Loading model and generating sample poems...")
    print(f"{'='*60}")
    
    from src.load_kaggle_model import load_kaggle_model
    
    try:
        model = load_kaggle_model(str(repo_root / "models" / "kaggle_trained_model"))
        model.load_tokenizer()
        
        print("\n Model loaded successfully!")
        print("\n Generating sample poems...\n")
        
        prompts = [
            "<POETRY>",
            "<POETRY> The moon",
            "<POETRY> In silence"
        ]
        
        for prompt in prompts:
            poems = model.generate_poem(
                prompt=prompt,
                max_length=120,
                num_return_sequences=1
            )
            print(f"\n--- Generated from '{prompt}' ---")
            print(poems[0])
            print("-" * 50)
        
        print("\nSetup complete! Model is ready to use.")
        print("\nNext steps:")
        print("1. Import in Python: from src.load_kaggle_model import load_kaggle_model")
        print("2. Load model: model = load_kaggle_model()")
        print("3. Generate: poems = model.generate_poem(prompt='<POETRY>', max_length=150)")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
