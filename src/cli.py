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


@cli.command()
def interactive():
    """Launch interactive haiku generator with GPT-Neo-1.3B"""
    click.echo("Launching interactive haiku generator...")
    click.echo("="*70)
    
    # Import and run the interactive haiku generator
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    
    # Import and launch
    try:
        from src.pretrained_models import load_pretrained_model
        from src.simple_haiku import generate_simple_haiku
        
        click.echo("\nChoose generator:")
        click.echo("1. Rule-based (fast, reliable, always coherent)")
        click.echo("2. AI Model (GPT-Neo-1.3B, slower, experimental)")
        
        choice = click.prompt("Select", type=str, default='1')
        
        if choice == '2':
            click.echo("\n[INFO] Loading GPT-Neo-1.3B model...")
            click.echo("[INFO] This may take a moment on first run...")
            
            try:
                model = load_pretrained_model('gpt-neo-1.3b')
                use_ai = True
                click.echo("[SUCCESS] Model loaded!\n")
            except Exception as e:
                click.echo(f"\n[WARNING] Could not load AI model: {e}")
                click.echo("[INFO] Using rule-based generator instead\n")
                use_ai = False
        else:
            click.echo("\n[INFO] Using fast rule-based generator\n")
            use_ai = False
            use_ai = False
        
        # Interactive loop
        while True:
            click.echo("="*70)
            prompt = click.prompt("Enter haiku topic (or 'quit' to exit)", default="nature")
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                click.echo("\nThank you for using Poetron!")
                break
            
            num_haikus = click.prompt("Number of haikus", default=1, type=int)
            num_haikus = max(1, min(3, num_haikus))
            
            if use_ai:
                temperature = click.prompt("Creativity (0.1-1.0)", default=0.8, type=float)
                temperature = max(0.1, min(1.0, temperature))
            
            click.echo(f"\n[INFO] Generating {num_haikus} haiku(s) about '{prompt}'...")
            click.echo("="*70)
            
            try:
                if use_ai:
                    haikus = model.generate_haiku(
                        prompt=prompt,
                        temperature=temperature,
                        num_return_sequences=num_haikus,
                        max_new_tokens=50
                    )
                else:
                    haikus = generate_simple_haiku(prompt, num_haikus)
                
                for i, haiku in enumerate(haikus, 1):
                    click.echo(f"\nHaiku {i}:")
                    click.echo("-" * 50)
                    click.echo(haiku)
                    click.echo("-" * 50)
                
                # Save option
                if click.confirm("\nSave haiku to file?", default=False):
                    filename = click.prompt("Filename", default="haiku.txt")
                    with open(filename, 'a', encoding='utf-8') as f:
                        f.write(f"\nTopic: {prompt}\n")
                        for i, haiku in enumerate(haikus, 1):
                            f.write(f"\nHaiku {i}:\n{haiku}\n")
                        f.write("\n" + "="*50 + "\n")
                    click.echo(f"Saved to {filename}")
                    
            except Exception as e:
                click.echo(f"\n[ERROR] Generation failed: {e}")
                
    except KeyboardInterrupt:
        click.echo("\n\nInterrupted. Goodbye!")
    except Exception as e:
        click.echo(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    cli()