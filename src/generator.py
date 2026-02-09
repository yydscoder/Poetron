import re

def enforce_haiku_structure(text):
    """
    Forces raw text into a 3-line Haiku-style format.
    """
    # Clean up artifacts and extra spaces
    text = re.sub(r'<.*?>', '', text) # Remove any remaining HTML-like tags
    text = " ".join(text.split())
    
    words = text.split()
    if len(words) < 3:
        return text

    # We manually wrap the words into a 3-line structure
    # Line 1: ~3-4 words
    # Line 2: ~5-6 words
    # Line 3: ~3-4 words
    line1 = " ".join(words[:4])
    line2 = " ".join(words[4:10])
    line3 = " ".join(words[10:15])

    haiku = f"{line1.capitalize()}\n{line2}\n{line3}"
    return haiku.strip()

def generate_poem(style, seed, model_path="models/poetry_model"):
    from load_kaggle_model import load_kaggle_model
    
    # Load model and generate raw text
    poetry_model = load_kaggle_model(model_path)
    raw_output = poetry_model.generate_poem(prompt=seed)[0]
    
    # Post-process based on style
    if style.lower() == "haiku":
        return enforce_haiku_structure(raw_output)
    
    return raw_output
