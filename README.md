# Poetron - AI-Powered Poetry Generation System

Welcome to Poetron! This project generates beautiful poems in various styles using AI.

## Project Lifecycle

### 1. Setup and Installation

#### Quick Start (Recommended)
To set up and run the interactive poetry generator:

**On Linux/macOS:**
```bash
cd Poetron
bash quickstart.sh
```

**On Windows:**
```bash
cd Poetron
python setup_and_run.py
```

Both scripts will:
1. Install all required dependencies
2. Download the pre-trained model
3. Run the interactive poetry generator

**Note:** `quickstart.sh` is a bash script that works on Linux/macOS systems, while `setup_and_run.py` is a Python script that replicates the same functionality for Windows users.

**Important for Windows users:** Make sure you have Microsoft Visual C++ Redistributable installed before running the setup. Download it from: https://aka.ms/vs/17/release/vc_redist.x64.exe

#### Manual Setup
If you prefer to set up manually:

1. Install dependencies:
   ```bash
   pip install torch>=2.1.0 transformers>=4.44.0 peft>=0.11.0 tokenizers>=0.19.0 huggingface_hub>=0.25.1 click>=8.0.0 requests>=2.25.0 kagglehub>=0.2.0
   ```

2. Download the model:
   ```bash
   # On Linux/macOS:
   bash download_kaggle_trained_model.sh
   
   # On Windows:
   # Run PowerShell as Administrator and execute:
   # Invoke-WebRequest -Uri "https://www.kaggle.com/api/v1/datasets/download/xongkoro/flavourtownpoetrongeneratormodel" -OutFile "$env:USERPROFILE\Downloads\flavourtownpoetrongeneratormodel.zip"
   # Then extract the zip file to models/kaggle_trained_model/
   ```

3. Run the interactive mode:
   ```bash
   python interactive_poet.py
   ```

### 2. Usage

#### Interactive Mode
Run the interactive poetry generator:

```bash
python interactive_poet.py
```

#### CLI Commands
Once the system is set up, you can use these commands:

1. Generate a haiku:
   ```bash
   python poetry_cli.py generate --style haiku --seed 'moonlight'
   ```

2. Generate a sonnet:
   ```bash
   python poetry_cli.py generate --style sonnet --seed 'love'
   ```

3. Generate free verse:
   ```bash
   python poetry_cli.py generate --style freeverse --seed 'ocean'
   ```

4. Export to file:
   ```bash
   python poetry_cli.py generate --style haiku --export
   ```

5. List available styles:
   ```bash
   python poetry_cli.py list-styles
   ```

#### API Integration
To use API refinement, set your API key as an environment variable:

**On Linux/macOS:**
```bash
export POETRON_API_KEY="your-api-key-here"
```

**On Windows:**
```cmd
set POETRON_API_KEY=your-api-key-here
```

Or enter it when prompted during setup.

### 3. Testing
Run the comprehensive test suite to verify everything works:

```bash
python test_project.py
```

### 4. Maintenance

#### Updating Dependencies
If you need to update dependencies, run:
```bash
pip install --upgrade torch transformers peft tokenizers huggingface_hub click requests kagglehub
```

#### Re-downloading Models
If you need to re-download the model:

**On Linux/macOS:**
```bash
bash download_kaggle_trained_model.sh
```

**On Windows:**
```cmd
python setup_and_run.py
```

### 5. Uninstall

To completely remove Poetron from your system:

```bash
python uninstall.py
```

This will remove:
- All installed Python dependencies
- Model files and directories
- Virtual environment (if created)
- .env file (if exists)

### 6. Troubleshooting

If you encounter issues during setup:
1. Make sure you have Python 3.7+ installed
2. Ensure you have sufficient disk space (at least 1.5GB)
3. Check your internet connection
4. Run the setup script again

Common issues and solutions:
- **Dependency installation fails**: Try installing packages individually
- **Model download fails**: Check your internet connection and try again
- **Module not found errors**: Make sure you're running from the Poetron directory
- **Permission errors on Windows**: Run Command Prompt or PowerShell as Administrator
- **DLL load failure on Windows**: Install Microsoft Visual C++ 2015-2022 Redistributable from https://aka.ms/vs/17/release/vc_redist.x64.exe
  - **Important**: PyTorch specifically requires the 2015-2022 version. If you have older versions (like 2010 or 2012), you still need to install the newer version.
  - After installation, restart your computer before running the poetry generator.

For additional help, run the test suite:
```bash
python test_project.py
```

## Requirements

- Python 3.7+
- At least 1.5GB of free disk space
- Internet connection for initial setup
- **Windows users only**: Microsoft Visual C++ Redistributable (download from: https://aka.ms/vs/17/release/vc_redist.x64.exe)
  - **Note**: PyTorch requires Visual C++ 2015-2022 Redistributable. Older versions (2010, 2012) are not sufficient.

## Features

- Generate poems in multiple styles (haiku, sonnet, freeverse)
- Customize temperature and length parameters
- Export poems to files
- API refinement (optional)
- Interactive mode for easy use
- Command-line interface for advanced users