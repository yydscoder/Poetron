"""
Poetry generation functionality for the Poetry Generator
"""

import torch
import random
from pathlib import Path
import re
import os

try:
    from .utils import format_poem_for_style
    from .refiner import refine_with_api
except ImportError:
    from utils import format_poem_for_style
    from refiner import refine_with_api


def generate_poem(
    style: str,
    seed: str = "",
    length: int = 50,
    model_path: str = "models/kaggle_trained_model",
    temperature: float = 0.8,
    max_new_tokens: int = 100
):
    """
    Generate a poem in the specified style using LoRA model.
    Falls back to rule-based generation if model unavailable.

    Args:
        style (str): The style of poem to generate ('haiku', 'sonnet', 'freeverse')
        seed (str): Seed words or themes for the poem
        length (int): Desired length of the poem (for free verse)
        model_path (str): Path to the trained LoRA adapter
        temperature (float): Sampling temperature for generation
        max_new_tokens (int): Maximum number of new tokens to generate

    Returns:
        str: Generated poem
    """
    # Prepare the prompt with style token
    style_token = f"<POETRY>"
    prompt = f"{style_token} {seed}".strip() if seed.strip() else style_token

    try:
        # Try to load and use the LoRA model
        from load_kaggle_model import load_kaggle_model

        model_path_obj = Path(model_path)
        if model_path_obj.exists():
            print(f"🧠 Loading trained LoRA model from {model_path}...")

            # Load model (Body + Brain)
            poetry_model = load_kaggle_model(model_path)
            poetry_model.load_tokenizer()

            # Generate using the model
            poems = poetry_model.generate_poem(
                prompt=prompt,
                max_length=max_new_tokens,
                temperature=temperature,
                num_return_sequences=1,
                style=style,  # Pass the style to influence generation parameters
                max_new_tokens=max_new_tokens
            )

            generated_text = poems[0] if poems else generate_fallback_poem(style, seed)

            # Extract the actual poem by removing the prompt part
            # Find where the actual poem starts after the prompt
            if prompt in generated_text:
                raw_poem = generated_text[len(prompt):].strip()
            else:
                raw_poem = generated_text.strip()

            # Clean up the raw poem to remove special characters and control characters
            import re
            # Remove control characters (except newlines and tabs)
            cleaned_raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', raw_poem)
            # Remove extra whitespace and normalize line breaks
            cleaned_raw = re.sub(r'\r\n', '\n', cleaned_raw)
            cleaned_raw = re.sub(r'\r', '\n', cleaned_raw)
            cleaned_raw = re.sub(r'\n+', '\n', cleaned_raw)

            # Remove the "seed style:" part that appears in the output
            # This pattern appears to be "trees haiku:" or similar
            cleaned_raw = re.sub(r'\w+\s+' + style + r':?\s*', '', cleaned_raw, flags=re.IGNORECASE)

            cleaned_raw = cleaned_raw.strip()

            # Apply style formatting
            formatted_raw = format_poem_for_style(cleaned_raw, style)
            # Conditionally refine via API if POETRON_API_KEY is provided
            api_key = os.getenv('POETRON_API_KEY')
            if api_key:
                try:
                    refined_poem = refine_with_api(formatted_raw, style, seed)
                    # If refinement returned an error string, fall back to formatted_raw
                    if isinstance(refined_poem, str) and refined_poem.startswith("Error:"):
                        return formatted_raw
                    return refined_poem
                except Exception:
                    return formatted_raw

            return formatted_raw
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")

    except Exception as e:
        # Fallback to rule-based generation
        print(f"📝 Using fallback poem generator ({str(e)[:50]}...)")
        return generate_fallback_poem(style, seed)


def generate_fallback_poem(style: str, seed: str = ""):
    """
    Generate a fallback poem if the model fails.

    Args:
        style (str): The style of poem to generate
        seed (str): Seed words or themes for the poem

    Returns:
        str: Fallback poem for flagging
    """
    if style.lower() == 'haiku':
        lines = [
            seed if seed else "Silent morning dew",
            "Whispers to the awakening earth",
            "Sunlight paints the leaves"
        ]
        return "\n".join(lines[:3])

    elif style.lower() == 'sonnet':
        lines = [
            f"When {seed} fills the air with gentle thought," if seed else "When morning light breaks through the silent dawn,",
            "And birds begin their chorus sweet and clear,",
            "The world awakens from night's peaceful sleep,",
            "As flowers bloom beneath the sky so blue.",
            "",
            "The gentle breeze carries a whispered prayer,",
            "Of love that grows like vines along the wall,",
            "While memories dance in sunlit summer air,",
            "And time moves slow, yet swift as eagles fly.",
            "",
            "So moments pass like clouds across the sky,",
            "Yet linger in the heart's most sacred space,",
            "Where dreams take flight and never say goodbye,",
            "To beauty found in time's eternal grace."
        ]
        return "\n".join(lines[:14])

    else:  # freeverse
        lines = [
            seed if seed else "The world breathes",
            "In rhythms unknown",
            "To those who rush",
            "Through life's maze",
            "",
            "But pause",
            "And listen",
            "To the silence",
            "Between heartbeats",
            "",
            "There lies",
            "The poetry",
            "Of existence"
        ]
        return "\n".join(lines)


def enforce_haiku_structure(text: str) -> str:
    """
    Attempt to format text as a haiku (3 lines with 5-7-5 syllable pattern).

    Note: This is a simplified approximation and doesn't perfectly count syllables.
    A more sophisticated approach would be needed for accurate syllable counting.

    Args:
        text (str): Input text to format as haiku

    Returns:
        str: Text formatted as haiku
    """
    lines = text.split('\n')
    haiku_lines = []

    # Take first three lines or create three lines from the text
    for i in range(3):
        if i < len(lines) and lines[i].strip():
            haiku_lines.append(lines[i].strip())
        else:
            # Create a line from remaining text
            words = ' '.join(lines[i:]).split() if i < len(lines) else text.split()
            if words:
                # Approximate 5-7-5 syllable pattern with word counts (5-7-5 words ≈ 5-7-5 syllables)
                word_counts = [5, 7, 5]
                line_words = words[:word_counts[i]] if i < len(word_counts) else words[:5]
                haiku_lines.append(' '.join(line_words))
                # Remove used words
                words = words[word_counts[i]:] if i < len(word_counts) else words[5:]

    return '\n'.join(haiku_lines[:3])


def generate_with_style_control(style: str, seed: str, model_path: str, temperature: float = 0.8):
    """
    Generate a poem with enhanced style control.

    Args:
        style (str): The style of poem to generate
        seed (str): Seed words or themes for the poem
        model_path (str): Path to the trained model
        temperature (float): Sampling temperature

    Returns:
        str: Generated poem with style control
    """
    # This would implement more sophisticated style control
    # For now, we will use the basic generation with post-processing
    poem = generate_poem(style, seed, model_path=model_path, temperature=temperature)

    if style.lower() == 'haiku':
        return enforce_haiku_structure(poem)

    return poem