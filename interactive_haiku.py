"""
Interactive Haiku Generator
"""
from src.simple_haiku import generate_simple_haiku
from src.pretrained_models import load_pretrained_model
import sys


def main():
    print("\n" + "="*70)
    print("  POETRON - Interactive Haiku Generator")
    print("="*70)
    
    # Ask which generator to use
    print("\nChoose generator:")
    print("1. Rule-based (fast, reliable, always coherent)")
    print("2. AI Model (GPT-Neo-1.3B, slower, experimental)")
    
    choice = input("\nSelect [1/2, default=1]: ").strip()
    
    if choice == '2':
        print("\n[INFO] Loading GPT-Neo-1.3B model...")
        print("[INFO] This is a 1.3 billion parameter model - first load may take a moment")
        print("[INFO] Model will be cached for faster subsequent loads")
        
        try:
            model = load_pretrained_model('gpt-neo-1.3b')
            print("[SUCCESS] Model loaded successfully!")
            use_ai = True
        except Exception as e:
            print(f"\n[ERROR] Failed to load GPT-Neo model: {e}")
            print("[INFO] Falling back to rule-based generator")
            use_ai = False
    else:
        print("\n[INFO] Using fast rule-based generator")
        use_ai = False
    
    print("\n" + "="*70)
    print("Ready to generate haikus!")
    print("="*70)
    
    while True:
        print("\n" + "="*70)
        prompt = input("Enter haiku topic (or 'quit' to exit): ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using Poetron!")
            break
        
        if not prompt:
            prompt = 'nature'
            print(f"Using default topic: {prompt}")
        
        # Settings
        num_input = input("Number of haikus [1]: ").strip()
        try:
            num_haikus = int(num_input) if num_input else 1
            num_haikus = max(1, min(3, num_haikus))
        except:
            num_haikus = 1
        
        temp_input = input("Creativity (0.1-1.0) [0.8]: ").strip()
        try:
            temperature = float(temp_input) if temp_input else 0.8
            temperature = max(0.1, min(1.0, temperature))
        except:
            temperature = 0.8
        
        print(f"\n[INFO] Generating {num_haikus} haiku(s) about '{prompt}'...")
        print("="*70)
        
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
                print(f"\nHaiku {i}:")
                print("-" * 50)
                
                # Handle dictionary format (from AI model with validation)
                if isinstance(haiku, dict):
                    if not haiku['valid']:
                        # Show original AI output
                        print("AI Model Output (Invalid 5-7-5):")
                        print(haiku['original'])
                        print("\n" + "-" * 50)
                        print("Rules-Based Alternative (Valid 5-7-5):")
                        print(haiku['rules_based'])
                    else:
                        # Valid AI output
                        print(haiku['original'])
                else:
                    # Simple string format (from rule-based generator)
                    print(haiku)
                
                print("-" * 50)
            
            # Save option
            save = input("\nSave haiku to file? (y/n) [n]: ").strip().lower()
            if save == 'y':
                filename = input("Filename [haiku.txt]: ").strip() or "haiku.txt"
                with open(filename, 'a', encoding='utf-8') as f:
                    f.write(f"\nTopic: {prompt}\n")
                    for i, haiku in enumerate(haikus, 1):
                        f.write(f"\nHaiku {i}:\n")
                        if isinstance(haiku, dict):
                            if not haiku['valid']:
                                f.write(f"AI Output (Invalid):\n{haiku['original']}\n\n")
                                f.write(f"Rules-Based Alternative:\n{haiku['rules_based']}\n")
                            else:
                                f.write(f"{haiku['original']}\n")
                        else:
                            f.write(f"{haiku}\n")
                    f.write("\n" + "="*50 + "\n")
                print(f"Saved to {filename}")
        
        except Exception as e:
            print(f"\n[ERROR] Generation failed: {e}")
            print("Try a different topic or use the rule-based generator (option 1)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
