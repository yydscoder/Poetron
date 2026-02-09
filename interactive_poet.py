#!/usr/bin/env python3
"""
Interactive Poetry Generator - Create poems with your trained model
"""

import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from poetry_generator import generate_poem
from utils import validate_style, export_poem


def main():
    """Interactive poetry generation interface."""
    print("Welcome to Poetron - AI-Powered Poetry Generator!")
    print("=" * 60)
    print("I can help you create beautiful poems in various styles.")
    print()
    
    # Define available styles
    styles = {
        '1': 'haiku',
        '2': 'sonnet', 
        '3': 'freeverse'
    }
    
    while True:
        print("Available poem styles:")
        for num, style in styles.items():
            if style == 'haiku':
                print(f"  {num}) {style.title()} - 3 lines with approximately 5-7-5 syllable pattern")
            elif style == 'sonnet':
                print(f"  {num}) {style.title()} - 14 lines with traditional structure")
            elif style == 'freeverse':
                print(f"  {num}) {style.title()} - Free-form poetry with expressive imagery")
        
        print("  4) Exit")
        print()
        
        choice = input("Select a style (1-4): ").strip()
        
        if choice == '4':
            print("\nThank you for using Poetron! Goodbye!")
            break
        elif choice in styles:
            selected_style = styles[choice]
            break
        else:
            print("\nInvalid choice. Please select 1, 2, 3, or 4.\n")
    
    if choice == '4':
        return
    
    print(f"\nSelected style: {selected_style.title()}")
    print()
    
    # Get the theme/seed
    seed = input(f"What would you like your {selected_style} to be about? (e.g., 'love', 'nature', 'ocean'): ").strip()
    
    if not seed:
        seed = "life"  # Default seed if none provided
        print(f"No theme provided, using default: '{seed}'")
    
    # Suggest sensible defaults per style
    style_defaults = {
        'haiku': {'temperature': 0.35, 'max_new_tokens': 40},
        'sonnet': {'temperature': 0.45, 'max_new_tokens': 200},
        'freeverse': {'temperature': 0.6, 'max_new_tokens': 200}
    }

    defaults = style_defaults.get(selected_style, {'temperature': 0.6, 'max_new_tokens': 100})

    print(f"\nGenerating your {selected_style} about '{seed}'...")
    print()

    # Ask user for optional tuning parameters
    try:
        temp_input = input(f"Temperature (sampling) [{defaults['temperature']}]: ").strip()
        temperature = float(temp_input) if temp_input else defaults['temperature']
    except Exception:
        temperature = defaults['temperature']

    try:
        tokens_input = input(f"Max new tokens [{defaults['max_new_tokens']}]: ").strip()
        max_new_tokens = int(tokens_input) if tokens_input else defaults['max_new_tokens']
    except Exception:
        max_new_tokens = defaults['max_new_tokens']
    
    print()
    print()
    
    try:
        # Generate the poem using the local model
        poem = generate_poem(
            style=selected_style,
            seed=seed,
            model_path="./models/poetry_model",
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )
        
        print("Your poem is ready!")
        print("=" * 50)
        print(poem)
        print("=" * 50)
        print()
        
        # Ask if user wants to export
        export_choice = input("Would you like to save this poem to a file? (y/n): ").strip().lower()
        if export_choice in ['y', 'yes', 'ye']:
            filename = export_poem(poem, selected_style)
            print(f"\n Poem saved to: {filename}")
        
        print("\nThank you for using Poetron! Would you like to create another poem?")
        another = input("(y/n): ").strip().lower()
        if another in ['y', 'yes', 'ye']:
            print()
            main()  # Recursive call to start over
        else:
            print("\nThank you for using Poetron! Goodbye!")
    
    except Exception as e:
        print(f"\nAn error occurred while generating your poem: {str(e)}")
        print("Please try again or contact support if the issue persists.")


if __name__ == '__main__':
    main()