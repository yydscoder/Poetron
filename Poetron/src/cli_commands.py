"""
Functions for CLI commands
"""

from poetry_generator import generate_poem as local_generate_poem
from api_integration import generate_poem_via_api


def generate_poem(style, seed, length, api, model_path, api_model='gpt2'):
    """
    Generate a poem based on the specified parameters.

    Args:
        style (str): The style of poem to generate ('haiku', 'sonnet', 'freeverse')
        seed (str): Seed words or themes for the poem
        length (int): Maximum length of the poem (for free verse)
        api (bool): Whether to use API for generation
        model_path (str): Path to the trained model
        api_model (str): Model to use for API generation

    Returns:
        str: Generated poem
    """
    if api:
        return generate_poem_via_api(style, seed, length, api_model=api_model)
    else:
        return local_generate_poem(style, seed, length, model_path)


def train_model(data_path, epochs, model_name, output_dir):
    """
    Train or fine-tune the poetry generation model.

    Args:
        data_path (str): Path to the training data file
        epochs (int): Number of training epochs
        model_name (str): Base model name for training
        output_dir (str): Directory to save the trained model

    Returns:
        str: Path to the trained model
    """
    from trainer import train_model as train_func
    return train_func(data_path, epochs, model_name, output_dir)