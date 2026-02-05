# Poetron - AI-Powered Poetry Generator

Poetron is an AI-powered poetry generation system that uses a fine-tuned GPT-2 model to create original poems in various styles. Experience the magic of AI-generated poetry through an intuitive interactive interface.

## Features

- Fine-tuned local GPT-2 model trained on poetry datasets from kaggle
- Generate poems in different styles: haiku, sonnet, and free verse
- Interactive command-line interface for seamless poetry creation
- Save your favorite poems to files

## Quick Start

Get started with Poetron in just two steps:

### 1. Run the Setup Script

Open your terminal in the project's root directory and run:

```bash
bash quickstart.sh
```

This automated script will:
- Set up a Python virtual environment
- Install all necessary dependencies (`torch`, `transformers`, etc.)
- Download the pre-trained poetry model from Kaggle
- Verify the setup with a test generation

### 2. Launch Interactive Mode

Once setup is complete, start creating poetry:

```bash
cd Poetron
python interactive_poet.py
```

## Using Interactive Mode

The interactive mode provides a friendly, guided experience:

1. **Select Your Style**: Choose from haiku, sonnet, or free verse
2. **Enter Your Theme**: Provide a topic or theme for your poem (e.g., "ocean", "love", "stars")
3. **Receive Your Poem**: The AI generates an original poem based on your selections
4. **Save (Optional)**: Choose to save your poem to a file in the `outputs/` directory
5. **Create More**: Generate additional poems or exit when finished

### Example Session

```
Welcome to Poetron - AI-Powered Poetry Generator!
============================================================
I can help you create beautiful poems in various styles.

Available poem styles:
  1) Haiku - 3 lines with approximately 5-7-5 syllable pattern
  2) Sonnet - 14 lines with traditional structure
  3) Freeverse - Free-form poetry with expressive imagery
  4) Exit

Select a style (1-4): 1

Selected style: Haiku

What would you like your haiku to be about? (e.g., 'love', 'nature', 'ocean'): sunset

Generating your haiku about 'sunset'...

Your poem is ready!
==================================================
Golden rays descend
Silent whispers paint the sky
Day bids night hello
==================================================

Would you like to save this poem to a file? (y/n): y

Poem saved to: outputs/haiku_20260205_143022.txt
```

## Manual Installation

If you prefer manual setup:

1. Navigate to the Poetron directory:
   ```bash
   cd Poetron
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the pre-trained model:
   ```bash
   bash download_kaggle_trained_model.sh
   ```

## Troubleshooting

- **Model not found**: Ensure the `download_kaggle_trained_model.sh` script completed successfully
- **Import errors**: Verify that your virtual environment is activated and all dependencies are installed
- **Generation errors**: Check that the model files are present in `models/poetry_model/`

## About the Model

Poetron uses a GPT-2 model fine-tuned on poetry datasets, enabling it to generate creative and contextually relevant poems across multiple styles. The model understands poetic structure, rhythm, and imagery to create original compositions.

Limitations - There is a limitation since GPT2 isnt very good at indexing data and may come out with related, but not grammatically correct outputs which is what the hackclub API is for, using it is reccomended but optional 

Enjoy creating beautiful poetry with Poetron! 
