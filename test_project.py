#!/usr/bin/env python3
"""
Comprehensive test suite for the Poetron poetry generation system
Run: python test_project.py
"""
import sys
import subprocess
from pathlib import Path

def run_test(name, command, description=""):
    """Run a test and report results"""
    print(f"\n{'='*70}")
    print(f"🧪 TEST: {name}")
    if description:
        print(f"📝 {description}")
    print(f"{'='*70}")
    print(f"Running: {command}\n")
    
    result = subprocess.run(command, shell=True, cwd=str(Path(__file__).parent / "Poetron"))
    
    if result.returncode == 0:
        print(f"✅ {name} PASSED")
        return True
    else:
        print(f"❌ {name} FAILED")
        return False

def main():
    print("\n" + "="*70)
    print("🎭 POETRON - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    results = {}
    
    # Test 1: Installation & Dependencies
    results["Dependencies"] = run_test(
        "Check Dependencies",
        "python -c \"import torch; import transformers; import click; print('✓ All dependencies installed')\"",
        "Verify required packages are installed"
    )
    
    # Test 2: Data Preprocessing
    results["Data Preprocessing"] = run_test(
        "Data Preprocessing Module",
        "python -c \"from src.data_preprocessing import load_poetry_data, clean_poem_text, add_style_tokens; print('✓ Data preprocessing module loaded')\"",
        "Load and test data preprocessing functions"
    )
    
    # Test 3: Poetry Generator Module
    results["Poetry Generator"] = run_test(
        "Poetry Generator Module",
        "python -c \"from src.poetry_generator import generate_fallback_poem; poem = generate_fallback_poem('test'); print(f'✓ Generated fallback poem: {len(poem)} chars')\"",
        "Test fallback poem generation"
    )
    
    # Test 4: Utils Module
    results["Utils"] = run_test(
        "Utils Module",
        "python -c \"from src.utils import validate_style, format_poem_for_style; print(f'✓ Valid styles: {validate_style(\\\"haiku\\\")}, {validate_style(\\\"sonnet\\\")}, {validate_style(\\\"freeverse\\\")}')\"",
        "Test utility functions"
    )
    
    # Test 5: CLI Module
    results["CLI Module"] = run_test(
        "CLI Module",
        "python poetry_cli.py --version",
        "Check CLI loads and shows version"
    )
    
    # Test 6: CLI Help
    results["CLI Help"] = run_test(
        "CLI Help Commands",
        "python poetry_cli.py list-styles",
        "List available poem styles"
    )
    
    # Test 7: Generate Fallback Poem
    results["Generate Fallback"] = run_test(
        "Generate Fallback Poem",
        "python poetry_cli.py generate --style haiku --seed 'morning'",
        "Generate a poem using fallback generator (no model needed)"
    )
    
    # Test 8: Load Kaggle Model Module
    results["Kaggle Model Loader"] = run_test(
        "Kaggle Model Loader Module",
        "python -c \"from src.load_kaggle_model import load_kaggle_model; print('✓ Kaggle model loader module loaded')\"",
        "Test Kaggle model loader imports"
    )
    
    # Test 9: Unit Tests
    results["Unit Tests"] = run_test(
        "Unit Tests",
        "python -m pytest tests/ -v || python -m unittest discover tests/",
        "Run all unit tests"
    )
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*70)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Project is ready to use.\n")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. See errors above.\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
