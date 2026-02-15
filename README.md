# Poetron - AI Haiku Generator

A Python-based haiku generator using GPT-Neo-1.3B with intelligent fallback to rule-based generation. Creates authentic 5-7-5 haikus with automatic validation and quality reporting.

## Quick Start

```bash
python setup_and_run.py
```

That's it! The script will:
1. Check your Python version (3.8+ required)
2. Install all dependencies automatically
3. Download GPT-Neo-1.3B model (~5GB on first run)
4. Launch the interactive haiku generator

## Features

- **Dual-Generator System**: AI-powered GPT-Neo with rules-based fallback
- **Automatic Validation**: Checks 5-7-5 syllable patterns
- **Quality Reporting**: Shows both AI output and corrected alternatives
- **Interactive Interface**: Easy-to-use command-line interface
- **Save to File**: Export your haikus to text files
- **Adjustable Creativity**: Control generation temperature (0.1-1.0)

## Example Output

```
Enter haiku topic (or 'quit' to exit): ocean

Haiku 1:
--------------------------------------------------
AI Model Output (Invalid 5-7-5):
The ocean waves crash on shore
in the dark of night
stars shimmer above the sea

--------------------------------------------------
Rules-Based Alternative (Valid 5-7-5):
Silent ocean waits
watching waves drift in spring rain
grace fills the shore
--------------------------------------------------
```

## How It Works

### Dual-Generator Architecture

Poetron uses a two-tier generation system:

#### 1. Primary: GPT-Neo-1.3B AI Model
- **Model**: EleutherAI/gpt-neo-1.3B (1.3 billion parameters)
- **Method**: Few-shot prompting with haiku examples
- **Inference**: CPU-based (no GPU required)
- **Download Size**: ~5GB (cached after first run)
- **Output**: Creative, varied haikus

**Technical Details:**
- Uses causal language modeling with custom prompts
- Few-shot examples guide the model toward haiku structure
- Generates up to 60 new tokens per haiku
- Parameters: `temperature=0.1-1.0`, `top_p=0.9`, `top_k=40`

#### 2. Fallback: Rules-Based Generator
- **Method**: Template-based with curated word banks
- **Validation**: Always produces valid 5-7-5 structure
- **Features**: Grammar-aware, theme-detection, typo correction
- **Speed**: Instant generation

**Technical Details:**
- 6 themed word banks (spring, summer, autumn, winter, night, water)
- Automatic verb conjugation (singular/plural agreement)
- Pronoun selection (subjective/objective forms)
- Syllable-counted templates with 3 variations per line type

### Validation System

When GPT-Neo generates a haiku, Poetron:

1. **Extracts Content**: Removes instruction artifacts and formatting
2. **Counts Syllables**: Uses heuristic-based counter with 50+ exception words
3. **Validates Structure**: Checks for exact 5-7-5 pattern
4. **Reports Results**: 
   - If valid: Shows AI output only
   - If invalid: Shows both AI output + rules-based alternative

**Syllable Counter Features:**
- Exception dictionary for common words (e.g., 'beautiful': 3, 'ocean': 2)
- Silent 'e' detection
- -ed ending handling
- Punctuation stripping

### Why Both Generators?

**GPT-Neo Strengths:**
- Creative and varied output
- Natural language flow
- Contextually rich imagery
- Unpredictable (interesting) results

**GPT-Neo Limitations:**
- Not trained specifically on poetry
- Often produces invalid syllable counts
- May generate 6-7-7 or 8-5-8 patterns
- No guaranteed haiku structure

**Rules-Based Strengths:**
- Always valid 5-7-5 structure
- Perfect grammar and agreement
- Instant generation
- Predictable quality

**Rules-Based Limitations:**
- Limited vocabulary (word banks)
- Templated structure
- Less creative variety
- Formulaic output

## Manual Setup (Optional)

If you prefer manual installation:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the generator
python interactive_haiku.py
```

## Requirements

- **Python**: 3.8 or higher
- **RAM**: 8GB+ recommended for GPT-Neo
- **Storage**: 6GB free space (for model cache)
- **Internet**: Required for first-time model download

## Dependencies

```
torch>=2.1.0         # PyTorch for model inference
transformers>=4.44.0 # Hugging Face transformers library
tokenizers>=0.19.0   # Fast tokenization
huggingface_hub      # Model downloading
click>=8.0.0         # CLI utilities
```

## Project Structure

```
Poetron/
├── setup_and_run.py           # Main entry point
├── interactive_haiku.py       # Interactive CLI interface
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── src/
    ├── simple_haiku.py        # Rules-based generator
    ├── pretrained_models.py   # GPT-Neo wrapper
    └── utils.py               # Helper functions
```

## Troubleshooting

### "Model download is slow"
- First download takes 5-10 minutes on slow connections
- Model is cached in `~/.cache/huggingface/` after first run
- Subsequent runs load instantly from cache

### "Out of memory error"
- GPT-Neo requires ~4GB RAM during inference
- Close other applications to free memory
- On low-RAM systems, use rules-based generator (option 1)

### "Invalid syllable counts"
- This is normal! GPT-Neo isn't trained on poetry
- The system shows you both versions automatically
- Use the rules-based alternative if you need strict 5-7-5

### "Import errors"
- Run: `pip install --upgrade -r requirements.txt`
- Ensure Python 3.8+ with: `python --version`

## Technical Specifications

### GPT-Neo-1.3B Configuration
- **Architecture**: Transformer decoder (causal LM)
- **Parameters**: 1.3 billion
- **Context Window**: 2048 tokens
- **Vocabulary**: 50,257 tokens
- **Precision**: FP32 (CPU inference)

### Generation Parameters
```python
{
    'max_new_tokens': 60,
    'temperature': 0.1-1.0,    # User adjustable
    'top_p': 0.9,
    'top_k': 40,
    'do_sample': True,
    'repetition_penalty': 1.2,
    'no_repeat_ngram_size': 2
}
```

### Syllable Counter Algorithm
```python
# Pseudocode
def count_syllables(word):
    1. Check exception dictionary
    2. Count vowel groups
    3. Adjust for silent 'e'
    4. Handle -ed endings
    5. Return max(1, count)
```

## Usage Examples

### Basic Generation
```bash
$ python setup_and_run.py
Enter haiku topic: mountain
Number of haikus [1]: 1
Creativity (0.1-1.0) [0.8]: 0.8
```

### High Creativity
```bash
Creativity: 0.9-1.0   # More random, experimental
```

### Conservative/Traditional
```bash
Creativity: 0.3-0.5   # More predictable, traditional
```

### Batch Generation
```bash
Number of haikus: 3   # Generate multiple variations
```

## Contributing

This project uses:
- **GPT-Neo-1.3B**: EleutherAI's open-source language model
- **Transformers**: Hugging Face's transformer library
- **PyTorch**: Deep learning framework

## License

MIT License - See LICENSE file for details

## Acknowledgments

- **EleutherAI**: for GPT-Neo-1.3B model
- **Hugging Face**: for transformers library
- **PyTorch Team**: for the deep learning framework

## Version

Current Version: 2.0
- Dual-generator system with intelligent fallback
- Automatic syllable validation
- Grammar-corrected rules-based generator
- Clean, professional interface

---

**Made with ❤️ for poetry enthusiasts and AI explorers**
