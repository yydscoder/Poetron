"""
Generate poems using the trained model
"""
import sys
from pathlib import Path
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
from utils import format_poem_for_style, export_poem


def load_model(model_path: str):
    """
    Load the trained model and tokenizer.
    
    Args:
        model_path (str): Path to the trained model
    
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model from {model_path}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    # Fix tokenizer padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set to eval mode
    model.eval()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Model loaded on device: {device}")
    return model, tokenizer, device


def generate_poem(
    model,
    tokenizer,
    device,
    prompt: str = "<POETRY>",
    max_length: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_poems: int = 1
):
    """
    Generate poems using the trained model.
    
    Args:
        model: Trained GPT-2 model
        tokenizer: GPT-2 tokenizer
        device: Torch device (cpu/cuda)
        prompt (str): Starting prompt (use <POETRY> prefix for best results)
        max_length (int): Maximum length of generated poem
        temperature (float): Sampling temperature (lower = more deterministic)
        top_p (float): Nucleus sampling parameter
        num_poems (int): Number of poems to generate
    
    Returns:
        list: Generated poems
    """
    # Ensure prompt includes poetry token for better results
    if not prompt.startswith('<POETRY>'):
        prompt = f"<POETRY> {prompt}"
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_poems,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            attention_mask=torch.ones_like(input_ids)
        )
    
    # Decode poems
    poems = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
    return poems


def main():
    """Main function to test poem generation."""
    # Load model
    model_path = '../../models/poetry_model/poetry_model_finetuned'
    model, tokenizer, device = load_model(model_path)
    
    print("\n" + "="*60)
    print("Poetry Generator - Interactive Mode")
    print("="*60 + "\n")
    
    while True:
        print("\nOptions:")
        print("1. Generate with default prompt")
        print("2. Generate with custom prompt")
        print("3. Generate multiple poems")
        print("4. Exit")
        
        choice = input("\nChoose an option (1-4): ").strip()
        
        if choice == '1':
            print("\nGenerating poem...")
            poems = generate_poem(
                model, tokenizer, device,
                prompt="<POETRY>",
                max_length=200,
                num_poems=1
            )
            
            for i, poem in enumerate(poems, 1):
                print(f"\n--- Generated Poem {i} ---")
                print(poem)
                print("-" * 40)
                
                # Ask if user wants to save
                save = input("Save this poem? (y/n): ").strip().lower()
                if save == 'y':
                    filepath = export_poem(poem, "generated")
                    print(f"Saved to: {filepath}")
        
        elif choice == '2':
            prompt = input("\nEnter a prompt (or press Enter for default): ").strip()
            if not prompt:
                prompt = "<POETRY>"
            else:
                # Ensure poetry token is added
                if not prompt.startswith('<POETRY>'):
                    prompt = f"<POETRY> {prompt}"
            
            print("\nGenerating poem...")
            poems = generate_poem(
                model, tokenizer, device,
                prompt=prompt,
                max_length=200,
                num_poems=1
            )
            
            for i, poem in enumerate(poems, 1):
                print(f"\n--- Generated Poem {i} ---")
                print(poem)
                print("-" * 40)
                
                save = input("Save this poem? (y/n): ").strip().lower()
                if save == 'y':
                    filepath = export_poem(poem, "generated")
                    print(f"Saved to: {filepath}")
        
        elif choice == '3':
            num = int(input("How many poems to generate? "))
            
            print(f"\nGenerating {num} poems...")
            poems = generate_poem(
                model, tokenizer, device,
                prompt="<POETRY>",
                max_length=200,
                num_poems=num
            )
            
            for i, poem in enumerate(poems, 1):
                print(f"\n--- Generated Poem {i} ---")
                print(poem)
                print("-" * 40)
        
        elif choice == '4':
            print("Goodbye!")
            break
        
        else:
            print("Invalid option. Please try again.")


if __name__ == '__main__':
    main()
