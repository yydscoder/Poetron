#!/usr/bin/env python3
"""
CLI interface for the Poetry Generator
"""

import click
import os
from pathlib import Path

# Add the src directory to the path so we can import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from cli_commands import generate_poem, train_model


@click.group()
@click.version_option(version='1.0.0')
def cli():
    """CLI Algorithmic Poetry Generator - Create algorithmic poems in multiple styles."""
    pass


@cli.command()
@click.option('--style', '-s', type=click.Choice(['haiku', 'sonnet', 'freeverse']),
              default='haiku', help='Poem style to generate')
@click.option('--seed', '-sd', default='', help='Seed words or themes for the poem')
@click.option('--length', '-l', default=50, type=int, help='Maximum length of the poem (for free verse)')
@click.option('--api', is_flag=True, help='Use API for generation instead of local model')
@click.option('--api-model', default='gpt2', help='Model to use for API generation (e.g., gpt2, facebook/opt-350m)')
@click.option('--export', is_flag=True, help='Export the poem to a file')
@click.option('--model-path', default='./models/poetry_model', help='Path to the trained local model')
def generate(style, seed, length, api, api_model, export, model_path):
    """Generate a poem in the specified style."""
    click.echo(f"Generating {style} poem with seed: '{seed}'")

    if api:
        click.echo(f"Using API with model: {api_model}")

    poem = generate_poem(style, seed, length, api, model_path, api_model)

    click.echo("\nGenerated Poem:")
    click.echo("="*50)
    click.echo(poem)
    click.echo("="*50)

    if export:
        from utils import export_poem
        filename = export_poem(poem, style)
        click.echo(f"\nPoem exported to {filename}")


@cli.command()
@click.option('--data', '-d', required=True, help='Path to the training data file')
@click.option('--epochs', '-e', default=3, type=int, help='Number of training epochs')
@click.option('--model-name', '-m', default='gpt2', help='Base model name for training')
@click.option('--output-dir', default='./models', help='Directory to save the trained model')
def train(data, epochs, model_name, output_dir):
    """Train or fine-tune the poetry generation model."""
    click.echo(f"Training model with data from {data}")
    click.echo(f"Using base model: {model_name}")
    click.echo(f"Epochs: {epochs}")
    
    model_path = train_model(data, epochs, model_name, output_dir)
    click.echo(f"Model trained and saved to {model_path}")


@cli.command()
def list_styles():
    """List all available poem styles."""
    styles = [
        "haiku - 3 lines with approximately 5-7-5 syllable pattern",
        "sonnet - 14 lines",
        "freeverse - User-defined length"
    ]
    
    click.echo("Available poem styles:")
    for style in styles:
        click.echo(f"  {style}")


if __name__ == '__main__':
    cli()