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
    # Clean up the text first - remove leading/trailing whitespace and common artifacts
    poem_text = poem_text.strip()

    # Remove common artifacts like leading commas or incomplete sentences
    if poem_text.startswith((',', '.', ';', ':', '-', '—')):
        # Find the first space and remove up to it
        space_idx = poem_text.find(' ')
        if space_idx != -1:
            poem_text = poem_text[space_idx+1:].strip()

    # Remove common internet artifacts and clean up encoding issues
    import re
    # Remove specific problematic terms like "RandomRedditor" but preserve legitimate words
    poem_text = re.sub(r'\bRandomRedditor\b', '', poem_text, flags=re.IGNORECASE)
    # Remove sequences that look like usernames (word with mixed case/digits that might be usernames)
    poem_text = re.sub(r'\b([A-Z][a-z]*[A-Z][a-z]*)\b', '', poem_text)  # Remove CamelCase words
    # Remove special characters that indicate encoding issues
    poem_text = re.sub(r'[^\x00-\x7F]+', '', poem_text)  # Remove non-ASCII characters
    # Clean up multiple spaces and empty areas
    poem_text = re.sub(r'\s+', ' ', poem_text).strip()
    # Remove standalone letters that might be artifacts
    poem_text = re.sub(r'\b[a-zA-Z]\b', '', poem_text).strip()

    lines = poem_text.split('\n')
    # Filter out empty lines
    lines = [line.strip() for line in lines if line.strip()]

    if style.lower() == 'haiku':
        # For haiku, we want to ensure exactly 3 lines
        # If we have 3 or more lines, take the first 3
        if len(lines) >= 3:
            result = '\n'.join(lines[:3])
        else:
            # If we have fewer than 3 lines, try to break them into 3
            text = ' '.join(lines)
            # Split by sentence endings or natural breaks
            # Split by periods, commas, or semicolons followed by space
            potential_breaks = re.split(r'[.!?]\s+|,\s+|;\s+', text)

            if len(potential_breaks) >= 3:
                # Take first 3 segments
                result = '\n'.join([seg.strip() for seg in potential_breaks[:3] if seg.strip()])
            elif len(potential_breaks) == 2:
                # If only 2 segments, try to split the longer one
                seg1 = potential_breaks[0].strip()
                seg2 = potential_breaks[1].strip()

                # If second segment is long, try to split it
                if len(seg2.split()) > 5:  # If more than 5 words
                    words = seg2.split()
                    mid_point = len(words) // 2
                    seg2_part1 = ' '.join(words[:mid_point])
                    seg2_part2 = ' '.join(words[mid_point:])
                    result = f"{seg1}\n{seg2_part1}\n{seg2_part2}"
                else:
                    result = f"{seg1}\n{seg2}"
            else:
                # If only one segment, try to split by word count
                words = text.split()
                if len(words) > 10:  # If more than 10 words, try to split
                    avg_per_line = len(words) // 3
                    line1 = ' '.join(words[:avg_per_line])
                    line2 = ' '.join(words[avg_per_line:avg_per_line*2])
                    line3 = ' '.join(words[avg_per_line*2:])
                    result = f"{line1}\n{line2}\n{line3}"
                else:
                    result = text

        # Ensure we have exactly 3 lines for haiku
        result_lines = result.split('\n')
        if len(result_lines) < 3:
            # Pad with empty lines if needed
            result_lines.extend([''] * (3 - len(result_lines)))
            result = '\n'.join(result_lines[:3])
        elif len(result_lines) > 3:
            # Trim to 3 lines if too many
            result = '\n'.join(result_lines[:3])

        return result
    elif style.lower() == 'sonnet':
        # Sonnet should be 14 lines
        return '\n'.join(lines[:14])
    else:
        # Free verse return as is
        return '\n'.join(lines)


def get_api_token():
    """
    Get the Hugging Face API token from environment variable.
    
    Returns:
        str: API token or None if not found
    """
    # Prefer POETRON_API_KEY (HackAI / Hack Club proxy)
    return os.getenv('POETRON_API_KEY')


def check_model_exists(model_path):
    """
    Check if a model exists at the specified path.
    
    Args:
        model_path (str): Path to check for model
    
    Returns:
        bool: True if model exists, False otherwise
    """
    return Path(model_path).exists()