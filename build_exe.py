"""
Build Script for Poetron Executable
Creates a standalone .exe for Windows distribution
"""
import subprocess
import sys
import os
from pathlib import Path
import shutil

def print_header(message):
    print("\n" + "="*70)
    print(f"  {message}")
    print("="*70)

def check_pyinstaller():
    """Check if PyInstaller is installed"""
    try:
        import PyInstaller
        print("[OK] PyInstaller is installed")
        return True
    except ImportError:
        print("[INFO] PyInstaller not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
            print("[OK] PyInstaller installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("[ERROR] Failed to install PyInstaller")
            return False

def build_exe():
    """Build the executable using PyInstaller"""
    print_header("Building Poetron Executable")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("[ERROR] Python 3.8 or higher is required")
        return False
    
    print(f"[OK] Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Check PyInstaller
    if not check_pyinstaller():
        return False
    
    # Clean previous builds
    print("\n[INFO] Cleaning previous builds...")
    for folder in ['build', 'dist']:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"[OK] Removed {folder}/")
    
    # Build using spec file
    print("\n[INFO] Building executable (this may take several minutes)...")
    print("[INFO] Please wait while PyInstaller packages the application...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "PyInstaller",
            "poetron.spec",
            "--clean"
        ])
        
        print_header("Build Successful!")
        print("\nYour executable is ready at: dist/Poetron/Poetron.exe")
        print("\nIMPORTANT NOTES:")
        print("1. The entire 'dist/Poetron' folder must be distributed together")
        print("2. Users still need to run setup once to download the GPT-Neo model")
        print("3. The model files (~5GB) are downloaded on first use")
        print("4. Distribute the whole folder or create an installer")
        print("\nTo test: cd dist/Poetron && Poetron.exe")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Build failed: {e}")
        return False

def create_release_package():
    """Create a release-ready package"""
    print("\n[INFO] Creating release package...")
    
    if not os.path.exists('dist/Poetron'):
        print("[ERROR] Build folder not found. Run build first.")
        return False
    
    # Copy README to dist
    readme_src = Path('README.md')
    if readme_src.exists():
        shutil.copy(readme_src, 'dist/Poetron/README.md')
        print("[OK] Copied README.md")
    
    # Create a quick start guide
    quick_start = """POETRON - Quick Start Guide
===========================

FIRST TIME USE:
1. Double-click Poetron.exe
2. Choose option 2 (AI Model) on first run to download GPT-Neo
3. Model download is ~5GB and happens once

SUBSEQUENT USE:
1. Double-click Poetron.exe
2. Choose your generator (1=Fast, 2=AI)
3. Enter topics and enjoy haikus!

REQUIREMENTS:
- Windows 10/11
- ~10GB free disk space (for model)
- Internet connection (first run only, for model download)

For full documentation, see README.md
"""
    
    with open('dist/Poetron/QUICK_START.txt', 'w') as f:
        f.write(quick_start)
    print("[OK] Created QUICK_START.txt")
    
    print("\n[SUCCESS] Release package ready in dist/Poetron/")
    return True

if __name__ == '__main__':
    print_header("Poetron Executable Builder")
    print("This script will create a standalone Windows executable")
    
    if build_exe():
        create_release_package()
        print("\n" + "="*70)
        print("  Build Complete!")
        print("="*70)
    else:
        print("\n[FAILED] Build process encountered errors")
        sys.exit(1)
