# Poetron - Poetry Generation System

Poetron is an AI-powered poetry generation system that uses transformer models to create original poems in various styles.

## Features

- Fine-tuned GPT-2 model on poetry datasets
- Generate poems in different styles (haiku, sonnet, free verse)
- Support for custom training data
- Command-line interface for easy usage

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd poetron
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Getting Started (After Cloning)

After cloning the repository, follow these steps to set up and run the project:

1. **Navigate to the project directory:**
```bash
cd Poetron
```

2. **Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install all required dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download the poetry dataset (optional but recommended):**
```bash
python download_data.py
```

5. **Create a sample dataset for faster training (optional):**
```bash
python src/sample_data.py
```

You're now ready to use Poetron! Proceed to the Usage section below.

## Usage

### Training the Model

To train the model on poetry data:

```bash
python poetry_cli.py train --data data/PoetryFoundationData.csv --epochs 3
```

### Generating Poems

To generate a poem:

```bash
python poetry_cli.py generate --style haiku --seed "morning dew"
```

Available styles: `haiku`, `sonnet`, `freeverse`

## Data

The system comes pre-configured to download and use the Poetry Foundation dataset from Kaggle. You can also use your own CSV file with columns for 'Poem', 'Title', etc.

To download the dataset manually:
```bash
python download_data.py
```

## Project Structure

- `poetry_cli.py` - Main command-line interface
- `src/` - Source code modules
  - `cli.py` - Command-line interface definitions
  - `trainer.py` - Model training functionality
  - `data_preprocessing.py` - Data loading and preprocessing
  - `poetry_generator.py` - Poem generation logic
- `data/` - Poetry datasets
- `models/` - Trained models
- `outputs/` - Generated poems

## Customization

You can customize the system by:
- Using your own poetry dataset in CSV format
- Adjusting training parameters (epochs, learning rate, etc.)
- Modifying the model architecture in the trainer
- Adding new poem styles to the generator
