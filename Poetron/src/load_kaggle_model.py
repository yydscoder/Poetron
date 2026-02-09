"""
Load and manage the Kaggle-trained poetry generator model
"""
import torch
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel

class KagglePoetryModel:
    def __init__(self, model_path: str = "models/kaggle_trained_model"):
        """
        Load the fine-tuned poetry model from Kaggle.

        Args:
            model_path: Path to the extracted Kaggle model directory
        """
        self.model_path = Path(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[INFO] Loading model from {model_path} on device: {self.device}")

        # Check if model path exists
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model path not found: {model_path}\n"
                f"Run: bash download_kaggle_trained_model.sh"
            )

        # Detect model type: merged or adapter-based
        self.is_adapter = (self.model_path / "adapter_config.json").exists()
        self.is_merged = (self.model_path / "config.json").exists() and not self.is_adapter

        if self.is_adapter:
            self._load_adapter_model()
        elif self.is_merged:
            self._load_merged_model()
        else:
            raise ValueError(
                f"Model structure not recognized in {model_path}\n"
                "Expected either adapter_config.json (LoRA) or config.json (merged)"
            )

    def _load_adapter_model(self):
        """Load base model + LoRA adapter (Brain + Body)"""
        print("📚 Loading base GPT-2 (the Body)...")
        # Load base model and tokenizer together to ensure they're aligned
        base_model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
        # Add special tokens that were used during training to match the vocabulary size
        special_tokens = {'additional_special_tokens': ['<POETRY>', '</POETRY>']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens)
        print(f"[INFO] Added {num_added_toks} special tokens to match trained model")
        
        # Resize model embeddings to accommodate the new tokens BEFORE loading the adapter
        base_model.resize_token_embeddings(len(tokenizer))
        print(f"[INFO] Resized model embeddings to {len(tokenizer)} to match trained model")

        print("[INFO] Attaching LoRA adapter from {}...".format(self.model_path))
        # Now load the adapter with the properly sized base model
        self.model = PeftModel.from_pretrained(base_model, str(self.model_path))

        print("[INFO] Moving to device...")
        self.model.to(self.device)
        self.model.eval()

        # Store the tokenizer as an instance variable
        self.tokenizer = tokenizer

        print("[SUCCESS] Your Poet is awake and ready!")

    def _load_merged_model(self):
        """Load merged fine-tuned model"""
        print("Loading merged GPT-2 model...")
        self.model = GPT2LMHeadModel.from_pretrained(str(self.model_path))
        self.model.to(self.device)
        print("✓ Merged model loaded")

    def load_tokenizer(self):
        """Load tokenizer and add custom poetry tags"""
        if hasattr(self, 'tokenizer'):
            # If tokenizer was already loaded with the model, return it
            return self.tokenizer
            
        print("📚 Loading tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add the custom tags you created during training
        print("[INFO] Adding special poetry tokens...")
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<POETRY>', '</POETRY>']})

        # Resize model embeddings to fit new tokens (only if model exists)
        if hasattr(self, 'model'):
            self.model.resize_token_embeddings(len(self.tokenizer))
            print(f"[SUCCESS] Tokenizer loaded (vocab size: {len(self.tokenizer)})")

        return self.tokenizer

    def generate_poem(
        self,
        prompt: str = "<POETRY>",
        max_length: int = 150,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
        style: str = "freeverse",  # Added style parameter to influence generation
        max_new_tokens: int = 100,  # Added max_new_tokens parameter
        min_new_tokens: int = 5,  # Added min_new_tokens parameter for compatibility
        **kwargs  # Accept any additional parameters for compatibility
    ) -> list:
        """
        Generate poems using the fine-tuned model.

        Args:
            prompt: Starting prompt (default: <POETRY> token)
            max_length: Max tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of poems to generate
            style: Style of poem to influence generation (for compatibility)
            max_new_tokens: Maximum new tokens to generate (for compatibility)
            min_new_tokens: Minimum new tokens to generate (for compatibility)
            **kwargs: Additional parameters for compatibility

        Returns:
            List of generated poem strings
        """
        if not hasattr(self, 'tokenizer'):
            self.load_tokenizer()

        self.model.eval()

        # Use the original prompt structure but apply style-specific post-processing
        # The model was trained on poetry, so it should generate poetry naturally
        if prompt.startswith("<POETRY>"):
            # If the prompt already has the tag, just add the content
            enhanced_prompt = f"{prompt} {style}: {prompt.replace('<POETRY>', '').strip()}"
        else:
            # Otherwise, add the tag and style info
            enhanced_prompt = f"<POETRY> {style}: {prompt}"

        input_ids = self.tokenizer.encode(enhanced_prompt, return_tensors='pt').to(self.device)

        with torch.no_grad():
            # Prepare generation arguments
            generation_kwargs = {
                'input_ids': input_ids,
                'max_length': min(max_length, len(input_ids[0]) + max_new_tokens),
                'num_return_sequences': num_return_sequences,
                'temperature': temperature,
                'top_p': top_p,
                'do_sample': True,
                'pad_token_id': self.tokenizer.eos_token_id,
                'attention_mask': torch.ones_like(input_ids),
                'repetition_penalty': 1.3,
                'no_repeat_ngram_size': 2,
            }
            
            # Add min_new_tokens if it's a valid parameter for this model
            if min_new_tokens > 0:
                generation_kwargs['min_length'] = len(input_ids[0]) + min_new_tokens

            outputs = self.model.generate(**generation_kwargs)

        poems = [
            self.tokenizer.decode(output, skip_special_tokens=True)  # Skip special tokens in output
            for output in outputs
        ]

        # Clean up the output to remove the prompt part
        cleaned_poems = []
        for poem in poems:
            # Remove the enhanced prompt from the beginning of the generated text
            if enhanced_prompt in poem:
                cleaned = poem[len(enhanced_prompt):].strip()
            else:
                cleaned = poem.strip()
            cleaned_poems.append(cleaned)

        return cleaned_poems

    def generate_batch(
        self,
        prompts: list,
        max_length: int = 150,
        temperature: float = 0.7
    ) -> dict:
        """
        Generate poems for multiple prompts.

        Args:
            prompts: List of prompt strings
            max_length: Max tokens per poem
            temperature: Sampling temperature

        Returns:
            Dictionary mapping prompts to generated poems
        """
        results = {}
        for prompt in prompts:
            poems = self.generate_poem(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=1
            )
            results[prompt] = poems[0]

        return results


def load_kaggle_model(model_path: str = "models/kaggle_trained_model") -> KagglePoetryModel:
    """Convenience function to load the Kaggle model"""
    return KagglePoetryModel(model_path)


if __name__ == "__main__":
    # Quick test
    model = load_kaggle_model()
    model.load_tokenizer()

    print("\n" + "="*60)
    print("GENERATING POEMS")
    print("="*60)

    # Generate some poems
    poems = model.generate_poem(
        prompt="<POETRY> trees",
        max_length=150,
        num_return_sequences=3,
        style="haiku"
    )

    for i, poem in enumerate(poems, 1):
        print(f"\n--- Poem {i} ---")
        print(poem)
        print("-" * 40)