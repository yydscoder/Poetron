"""
Poetry generation functionality for the Poetry Generator
"""

import torch
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path
import re

from trainer import load_trained_model
from utils import format_poem_for_style


def generate_poem(
    style: str, 
    seed: str = "", 
    length: int = 50, 
    model_path: str = "./models/poetry_model", 
    temperature: float = 0.8,
    max_new_tokens: int = 100
):
    """
    Generate a poem in the specified style.
    
    Args:
        style (str): The style of poem to generate ('haiku', 'sonnet', 'freeverse')
        seed (str): Seed words or themes for the poem
        length (int): Desired length of the poem (for free verse)
        model_path (str): Path to the trained model
        temperature (float): Sampling temperature for generation
        max_new_tokens (int): Maximum number of new tokens to generate
        
    Returns:
        str: Generated poem
    """
    # Prepare the prompt with style token
    style_token = f"<{style.upper()}>"
    prompt = f"{style_token} {seed}".strip()
    
    # If no seed is provided, use a generic prompt for the stylem this is also for debugging to ensure that there are no crashes involved and if this specific poem were to pop up we would know that there is someting wrong 
    if not seed.strip():
        if style.lower() == 'haiku':
            prompt = f"{style_token} Morning dew glistens"
        elif style.lower() == 'sonnet':
            prompt = f"{style_token} Upon a time when love was pure and bright"
        else:  # freeverse
            prompt = f"{style_token} The world unfolds"
    
    try:
        # Load the trained model
        model, tokenizer = load_trained_model(model_path)
        
        # Encode the prompt
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                inputs, 
                max_length=len(inputs[0]) + max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.encode('\n')[0] if '\n' in tokenizer.get_vocab() else None
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the poem part (remove the prompt)
        poem_part = generated_text[len(prompt):].strip()
        
        # Format according to style
        formatted_poem = format_poem_for_style(poem_part, style)
        
        return formatted_poem
        
    except Exception as e:
        # Fallback response if model loading or generation fails
        print(f"Error generating poem: {e}")
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