# Poetron - AI-Powered Poetry Generation System

Welcome to Poetron! This project generates beautiful poems in various styles using AI.

## Project Lifecycle

### 1. Setup and Installation

#### Quick Start (Recommended)
To set up and run the interactive poetry generator:

```bash
cd Poetron
python setup_and_run.py
```

This script will:
1. Install all required dependencies
2. Download the pre-trained model
3. Run the interactive poetry generator

#### Manual Setup
If you prefer to set up manually:

1. Install dependencies:
   ```bash
   pip install torch>=2.1.0 transformers>=4.44.0 peft>=0.11.0 tokenizers>=0.19.0 huggingface_hub>=0.25.1 click>=8.0.0 requests>=2.25.0 kagglehub>=0.2.0
   ```

2. Download the model:
   ```bash
   bash download_kaggle_trained_model.sh
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
```bash
export POETRON_API_KEY="your-api-key-here"
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
```bash
bash download_kaggle_trained_model.sh
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

For additional help, run the test suite:
```bash
python test_project.py
```

## Requirements

- Python 3.7+
- At least 1.5GB of free disk space
- Internet connection for initial setup

## Features

- Generate poems in multiple styles (haiku, sonnet, freeverse)
- Customize temperature and length parameters
- Export poems to files
- API refinement (optional)
- Interactive mode for easy use
- Command-line interface for advanced users