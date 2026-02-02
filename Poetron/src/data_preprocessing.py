import json
import re
from pathlib import Path
from typing import List
import pandas as pd

def load_poetry_data(data_path: str) -> List[str]:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    ext = data_path.suffix.lower()
    if ext == '.csv':
        df = pd.read_csv(data_path)
        # Flexible column detection function 
        possible_cols = ['poem', 'text', 'content', 'verse', 'poetry']
        target_col = next((c for c in df.columns if c.lower() in possible_cols), None)
        if target_col:
            return df[target_col].dropna().astype(str).tolist()
    elif ext in ['.json', '.jsonl']:
        # Handle JSONL/JSON
        if ext == '.jsonl':
            with open(data_path, 'r') as f:
                return [json.loads(line).get('text', '') for line in f if line.strip()]
        else:
            with open(data_path, 'r') as f:
                data = json.load(f)
                return data if isinstance(data, list) else [str(data)]
    
    # Default to TXT split by double newline
    content = data_path.read_text(encoding='utf-8')
    return [p.strip() for p in content.split('\n\n') if p.strip()]

def clean_poem_text(text: str) -> str:
    # Preserve newlines but collapse horizontal whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    # Basic normalization
    text = re.sub(r'[`\']', "'", text)
    text = re.sub(r'[^\w\s.,!?;:\'"-]', '', text)
    return text.strip()

def add_style_tokens(poems: List[str], style: str) -> List[str]:
    """Fixed to handle lists correctly as expected by your trainer."""
    style_token = f"<{style.upper()}>"
    end_token = f"</{style.upper()}>"
    return [f"{style_token} {p} {end_token}" for p in poems]

def split_into_training_chunks(texts: List[str], max_length: int = 512) -> List[str]:
    """Optimized chunking to keep poems together where possible."""
    chunks = []
    for text in texts:
        if len(text) <= max_length:
            chunks.append(text)
        else:
            # Simple split if poem is too long
            for i in range(0, len(text), max_length):
                chunks.append(text[i:i+max_length])
    return chunks

def preprocess_poetry_data(data_path: str, style: str = "POETRY") -> List[str]:
    raw_poems = load_poetry_data(data_path)
    return [clean_poem_text(p) for p in raw_poems if len(p) > 10]