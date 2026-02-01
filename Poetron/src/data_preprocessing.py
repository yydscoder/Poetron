"""
Data preprocessing functions for the Poetry Generator
"""

import json
import csv
import re
from pathlib import Path
from typing import List, Dict, Union
import pandas as pd




def load_poetry_data(data_path: str) -> List[str]:
    """
    This is to load poetry data from various file formats.
    
    Arguments:
        data_path (str): Path to the data file
        
    Returns:
        List[str]: List of poems or poem texts
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    extension = data_path.suffix.lower()
    
    if extension == '.txt':
        return _load_txt_file(data_path)
    elif extension == '.csv':
        return _load_csv_file(data_path)
    elif extension in ['.json', '.jsonl']:
        return _load_json_file(data_path)
    else:
        # For now lets assume it's a text file since we are in the extremely early stages of the project
        return _load_txt_file(data_path)


def _load_txt_file(file_path: Path) -> List[str]:
    """Load poems from a text file."""
    content = file_path.read_text(encoding='utf-8')
    # Split by double newlines which typically separate poems
    poems = [poem.strip() for poem in content.split('\n\n') if poem.strip()]
    return poems


def _load_csv_file(file_path: Path) -> List[str]:
    """Load poems from a CSV file."""
    poems = []
    df = pd.read_csv(file_path)
    
    # Look for poem column (case-insensitive)
    poem_col = None
    for col in df.columns:
        if col.lower() == 'poem':
            poem_col = col
            break
    
    if poem_col is None:
        # Fallback: try common column names
        for col in ['text', 'content', 'verse', 'poetry']:
            if col in df.columns:
                poem_col = col
                break
    
    if poem_col:
        for poem_text in df[poem_col]:
            if pd.notna(poem_text):
                poem_text = str(poem_text).strip()
                if poem_text:
                    poems.append(poem_text)
    
    return poems


def _load_json_file(file_path: Path) -> List[str]:
    """Load poems from a JSON file."""
    poems = []
    content = file_path.read_text(encoding='utf-8')
    
    if file_path.suffix.lower() == '.jsonl':
        # JSONL format - one JSON object per line
        for line in content.splitlines():
            if line.strip():
                data = json.loads(line)
                # Look for common keys that might contain poem text
                for key in ['text', 'poem', 'content', 'verse']:
                    if key in data:
                        poem_text = data[key].strip()
                        if poem_text:
                            poems.append(poem_text)
                        break
    else:
        # Regular JSON format
        data = json.loads(content)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # Look for common keys that might contain poem text
                    for key in ['text', 'poem', 'content', 'verse']:
                        if key in item and item[key].strip():
                            poems.append(item[key].strip())
                            break
                elif isinstance(item, str) and item.strip():
                    poems.append(item.strip())
        elif isinstance(data, dict):
            # Look for common keys that might contain poem text
            for key in ['text', 'poem', 'content', 'verse']:
                if key in data:
                    if isinstance(data[key], list):
                        poems.extend([item.strip() for item in data[key] if item.strip()])
                    elif isinstance(data[key], str) and data[key].strip():
                        poems.append(data[key].strip())
                    break
    
    return poems


def clean_poem_text(text: str) -> str:
    """
    Clean and normalize poem text.
    
    Args:
        text (str): Raw poem text
        
    Returns:
        str: Cleaned poem text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize quotes
    text = re.sub(r'[`\'\"]', '"', text)
    
    # Remove special characters that might interfere with training
    text = re.sub(r'[^\w\s\n\r\t.,!?;:\'"-]', ' ', text)
    
    # Normalize line breaks
    text = re.sub(r'\r\n|\r', '\n', text)
    
    # Normalize spaces around punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,!?;:])\s*', r'\1 ', text)
    
    return text.strip()


def preprocess_poetry_data(data_path: str, style: str = None) -> List[str]:
    """
    Load and preprocess poetry data from CSV or text files.
    
    Args:
        data_path (str): Path to poetry data CSV file or directory
        style (str): Optional style to filter by
    
    Returns:
        List[str]: List of preprocessed poems
    """
    # Load raw poems
    raw_poems = load_poetry_data(data_path)
    
    # Clean and process poems
    poems = []
    for poem in raw_poems:
        cleaned = clean_poem_text(poem)
        if cleaned:
            if style:
                cleaned = add_style_tokens(cleaned, style)
            poems.append(cleaned)
    
    return poems


def split_into_training_chunks(texts: List[str], max_length: int = 512) -> List[str]:
    """
    Split texts into chunks of specified maximum length.
    
    Args:
        texts (List[str]): List of text strings to chunk
        max_length (int): Maximum length of each chunk
        
    Returns:
        List[str]: List of text chunks
    """
    chunks = []
    for text in texts:
        # Split text into sentences or phrases
        sentences = re.split(r'[.!?]+', text)
        
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Add space if needed
            if current_chunk and not current_chunk.endswith(' '):
                sentence = ' ' + sentence
            
            # If adding this sentence would exceed max length, start a new chunk
            if len(current_chunk) + len(sentence) > max_length:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += sentence
        
        # Add the remaining chunk if it exists
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
    
    # Filter out empty chunks
    return [chunk for chunk in chunks if chunk.strip()]


def add_style_tokens(text: str, style: str) -> str:
    """
    Add style tokens to a poem for conditional generation.
    
    Args:
        text (str): Single poem text
        style (str): Style to add tokens for
        
    Returns:
        str: Poem with style tokens added
    """
    style_token = f"<{style.upper()}>"
    end_token = f"</{style.upper()}>"
    return f"{style_token} {text} {end_token}"