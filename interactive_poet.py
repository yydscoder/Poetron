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
    
    # Define available styles to the user
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

    # Provide user-friendly explanations for parameters
    print("You can customize the creativity and length of your poem:")
    
    # Temperature explanation
    print(f"\nCreativity level (Temperature):")
    print(f"  - Low (0.1-0.3): More predictable, traditional style [{defaults['temperature']} is default for {selected_style}]")
    print(f"  - Medium (0.3-0.7): Balanced creativity and coherence")
    print(f"  - High (0.7-1.0): Very creative, unexpected expressions")
    
    try:
        temp_input = input(f"\nChoose creativity level (0.1-1.0) [{defaults['temperature']}]: ").strip()
        temperature = float(temp_input) if temp_input else defaults['temperature']
        
        # Validate temperature range
        if temperature < 0.1 or temperature > 1.0:
            print(f"Value out of range. Using default: {defaults['temperature']}")
            temperature = defaults['temperature']
    except Exception:
        temperature = defaults['temperature']

    # Max new tokens explanation
    print(f"\nPoem length (Max New Tokens):")
    if selected_style == 'haiku':
        print(f"  - Short (20-40): Traditional haiku style [{defaults['max_new_tokens']} is default)")
        print(f"  - Medium (40-80): Extended haiku with more detail")
        print(f"  - Long (80-150): Haiku-inspired poem with elaborate imagery")
    elif selected_style == 'sonnet':
        print(f"  - Standard (150-200): Traditional sonnet length [{defaults['max_new_tokens']} is default)")
        print(f"  - Extended (200-300): Longer sonnet with expanded themes")
    else:  # freeverse
        print(f"  - Brief (50-100): Short, concise poem [{defaults['max_new_tokens']} is default)")
        print(f"  - Standard (100-200): Medium-length expressive poem")
        print(f"  - Extended (200-400): Detailed, elaborate poem")

    try:
        tokens_input = input(f"\nChoose length (recommended: {defaults['max_new_tokens']}): ").strip()
        max_new_tokens = int(tokens_input) if tokens_input else defaults['max_new_tokens']
        
        # Validate token range based on style
        if selected_style == 'haiku':
            if max_new_tokens < 20:
                print("Too short for a haiku. Using minimum: 20")
                max_new_tokens = 20
            elif max_new_tokens > 150:
                print("Too long for a haiku. Using maximum: 150")
                max_new_tokens = 150
        elif selected_style == 'sonnet':
            if max_new_tokens < 100:
                print("Too short for a sonnet. Using minimum: 100")
                max_new_tokens = 100
            elif max_new_tokens > 400:
                print("Too long for a sonnet. Using maximum: 400")
                max_new_tokens = 400
        else:  # freeverse
            if max_new_tokens < 30:
                print("Too short for meaningful free verse. Using minimum: 30")
                max_new_tokens = 30
            elif max_new_tokens > 500:
                print("Too long. Using maximum: 500")
                max_new_tokens = 500
                
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