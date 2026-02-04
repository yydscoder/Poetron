# Poetron - Poetry Generation System

Poetron is an AI-powered poetry generation system that uses transformer models to create original poems in various styles. This project was analyzed and refactored to improve clarity and remove redundancies. Several irrelevant files were removed, including backup files, duplicate requirement files, and redundant scripts. The project structure was also simplified.

## Features

- Fine-tuned GPT-2 model on poetry datasets
- Generate poems in different styles (haiku, sonnet, free verse)
- Support for custom training data
- Command-line interface for easy usage
- API integration for enhanced poetry refinement (optional)

## How to Run Poetron From Scratch

The easiest way to get started with Poetron is by using the `quickstart.sh` script. This script will automate the entire setup process.

### Automated Setup (Recommended)

1.  **Run the Quickstart Script:**
    Open your terminal and run the following command from the project's root directory:

    ```bash
    bash quickstart.sh
    ```

    The script will:
    - Set up a virtual environment.
    - Install all necessary dependencies.
    - Download the pre-trained model.
    - Test the system with a sample poem.
    - Optionally configure an API key for enhanced refinement.

### Manual Installation

If you prefer to install manually, follow these steps:

1.  **Navigate to the project directory:**
    ```bash
    cd Poetron
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install all required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the pre-trained model:**
    ```bash
    bash download_kaggle_trained_model.sh
    ```

## Usage

### Generating Poems

To generate a poem:

```bash
python poetry_cli.py generate --style haiku --seed "YOUR PROMPT GOES HERE"
```

Available styles: `haiku`, `sonnet`, `freeverse`

### API Integration (Optional)

For enhanced poetry refinement, you can provide an API key during the quickstart process or by setting the `POETRON_API_KEY` environment variable. The system will automatically use the API to refine generated poems when available.

### Interactive Mode

For an interactive poetry creation experience:

```bash
python interactive_poet.py
```

### Training the Model

To train the model on your own poetry data:

```bash
python poetry_cli.py train --data path/to/your/data.csv --epochs 3
```

## Project Structure

- `poetry_cli.py` - Main command-line interface
- `interactive_poet.py` - Interactive poetry creation interface
- `src/` - Source code modules
  - `cli.py` - Command-line interface definitions
  - `cli_commands.py` - CLI command implementations
  - `trainer.py` - Model training functionality
  - `data_preprocessing.py` - Data loading and preprocessing
  - `poetry_generator.py` - Poem generation logic
  - `refiner.py` - API-based poem refinement
  - `utils.py` - Utility functions
- `models/` - Trained models
- `data/` - Poetry datasets (if added)
- `outputs/` - Generated poems

## Configuration

### API Key Setup

For enhanced post generation poetry refinement, you can provide an API key:

1. During the quickstart process
2. By setting an environment variable:
```bash
export POETRON_API_KEY="your-api-key-here"
```
3. By saving it to a `.env` file:
```bash
echo "POETRON_API_KEY=your-api-key-here" > .env
source .env
```

## Customization

You can customize the system by:
- Using your own poetry dataset in CSV format
- Adjusting training parameters (epochs, learning rate, etc.)
- Modifying the model architecture in the trainer
- Adding new poem styles to the generator
- Enhancing the refinement process in the refiner module
