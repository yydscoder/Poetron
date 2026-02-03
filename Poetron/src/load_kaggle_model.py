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
        
        print(f"🔧 Loading model from {model_path} on device: {self.device}")
        
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
        """Load base model + LoRA adapter"""
        print("Loading base GPT-2 + LoRA adapter...")
        base_model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.model = PeftModel.from_pretrained(base_model, str(self.model_path))
        self.model.to(self.device)
        print("✓ Adapter model loaded")
    
    def _load_merged_model(self):
        """Load merged fine-tuned model"""
        print("Loading merged GPT-2 model...")
        self.model = GPT2LMHeadModel.from_pretrained(str(self.model_path))
        self.model.to(self.device)
        print("✓ Merged model loaded")
    
    def load_tokenizer(self):
        """Load tokenizer from model directory"""
        print("Loading tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(str(self.model_path))
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"✓ Tokenizer loaded (vocab size: {len(self.tokenizer)})")
        return self.tokenizer
    
    def generate_poem(
        self,
        prompt: str = "<POETRY>",
        max_length: int = 150,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1
    ) -> list:
        """
        Generate poems using the fine-tuned model.
        
        Args:
            prompt: Starting prompt (default: <POETRY> token)
            max_length: Max tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of poems to generate
        
        Returns:
            List of generated poem strings
        """
        if not hasattr(self, 'tokenizer'):
            self.load_tokenizer()
        
        self.model.eval()
        
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        poems = [
            self.tokenizer.decode(output, skip_special_tokens=False)
            for output in outputs
        ]
        
        return poems
    
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
        prompt="<POETRY>",
        max_length=150,
        num_return_sequences=3
    )
    
    for i, poem in enumerate(poems, 1):
        print(f"\n--- Poem {i} ---")
        print(poem)
        print("-" * 40)
