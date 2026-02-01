"""
Utility functions for the Poetry Generator
"""

import os
from datetime import datetime
from pathlib import Path


def export_poem(poem, style):
    """
    Export the generated poem to a text file.
    
    Args:
        poem (str): The poem to export
        style (str): The style of the poem
    
    Returns:
        str: Path to the exported file
    """
    # Create outputs directory if it doesn't exist
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    # Generate a filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = outputs_dir / f"poem_{style}_{timestamp}.txt"
    
    # Write poem to file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Style: {style}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 50 + "\n\n")
        f.write(poem)
    
    return str(filename)


def validate_style(style):
    """
    Validate that the provided style is supported.
    
    Args:
        style (str): The style to validate
    
    Returns:
        bool: True if style is valid, False otherwise
    """
    valid_styles = ['haiku', 'sonnet', 'freeverse']
    return style.lower() in valid_styles


def format_poem_for_style(poem_text, style):
    """
    Format the poem according to the specified style.
    
    Args:
        poem_text (str): Raw poem text
        style (str): The style to format for
    
    Returns:
        str: Formatted poem
    """
    lines = poem_text.strip().split('\n')
    
    if style.lower() == 'haiku':
        # Haiku should usually be 3 lines, but we'll just ensure it's roughly the right length
        # Actual formatting will happen during generation
        return '\n'.join(lines[:3])
    elif style.lower() == 'sonnet':
        # Sonnet should be 14 lines
        return '\n'.join(lines[:14])
    else:
        # Free verse return as is
        return poem_text


def get_api_token():
    """
    Get the Hugging Face API token from environment variable.
    
    Returns:
        str: API token or None if not found
    """
    return os.getenv('HF_API_TOKEN')


def check_model_exists(model_path):
    """
    Check if a model exists at the specified path.
    
    Args:
        model_path (str): Path to check for model
    
    Returns:
        bool: True if model exists, False otherwise
    """
    return Path(model_path).exists()