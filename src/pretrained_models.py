"""
Pre-trained model loader for better poetry generation.
Supports multiple models optimized for creative text generation.
"""
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)
from typing import List, Optional
import re


class PretrainedPoetryModel:
    """Wrapper for pre-trained models optimized for poetry generation"""
    
    AVAILABLE_MODELS = {
        'gpt-neo-125m': {
            'name': 'EleutherAI/gpt-neo-125M',
            'type': 'causal',
            'description': 'Better than GPT-2, good for creative text'
        },
        'gpt-neo-1.3b': {
            'name': 'EleutherAI/gpt-neo-1.3B',
            'type': 'causal',
            'description': 'Larger model, better quality (requires more RAM)'
        },
        'distilgpt2': {
            'name': 'distilgpt2',
            'type': 'causal',
            'description': 'Faster, smaller GPT-2 variant'
        },
        'flan-t5-small': {
            'name': 'google/flan-t5-small',
            'type': 'seq2seq',
            'description': 'Instruction-following model, good for prompts'
        },
        'flan-t5-base': {
            'name': 'google/flan-t5-base',
            'type': 'seq2seq',
            'description': 'Better instruction following, larger'
        }
    }
    
    def __init__(self, model_name: str = 'gpt-neo-125m'):
        """Initialize with a pre-trained model
        
        Args:
            model_name: One of the keys from AVAILABLE_MODELS
        """
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model {model_name} not available. Choose from: {list(self.AVAILABLE_MODELS.keys())}")
        
        self.model_config = self.AVAILABLE_MODELS[model_name]
        self.model_name = self.model_config['name']
        self.model_type = self.model_config['type']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"[INFO] Loading {model_name} ({self.model_config['description']})...")
        print(f"[INFO] Device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.model_type == 'causal':
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"[SUCCESS] {model_name} loaded and ready!")
    
    def generate_haiku(
        self,
        prompt: str,
        temperature: float = 0.7,
        num_return_sequences: int = 1,
        max_new_tokens: int = 50
    ) -> List[str]:
        """Generate haiku poems using the pre-trained model
        
        Args:
            prompt: Topic or theme for the haiku
            temperature: Sampling temperature (0.1-1.0)
            num_return_sequences: Number of haikus to generate
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            List of generated haiku strings
        """
        if self.model_type == 'seq2seq':
            # For instruction-following models like T5
            instruction = (
                f"Write a haiku about {prompt}. "
                f"Three lines, 5-7-5 syllables:"
            )
            input_ids = self.tokenizer.encode(instruction, return_tensors='pt').to(self.device)
            attention_mask = torch.ones_like(input_ids)
        else:
            # For causal models - explicit haiku format with examples
            instruction = (
                f"Write haiku poems in 5-7-5 syllable format.\n\n"
                f"Topic: mountain\n"
                f"Snow caps the summit\n"
                f"ancient stones remember storms\n"
                f"silence speaks of time\n\n"
                f"Topic: river\n"
                f"Water flows gently\n"
                f"over smooth and weathered stones\n"
                f"journey never ends\n\n"
                f"Topic: {prompt}\n"
            )
            input_ids = self.tokenizer.encode(instruction, return_tensors='pt').to(self.device)
            attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=60,
                temperature=temperature,
                top_p=0.9,
                top_k=40,
                do_sample=True,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2
            )
        
        poems = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            # Extract the haiku without strict validation
            haiku = self._extract_haiku(text, prompt)
            if haiku:
                # Validate and report
                is_valid = self._validate_haiku_structure(haiku)
                if not is_valid:
                    print(f"[INFO] Generated haiku does not follow 5-7-5 syllable pattern")
                    # Show both versions
                    from src.simple_haiku import generate_simple_haiku
                    fallback = generate_simple_haiku(prompt, 1)[0]
                    poems.append({
                        'original': haiku,
                        'rules_based': fallback,
                        'valid': False
                    })
                else:
                    poems.append({
                        'original': haiku,
                        'rules_based': None,
                        'valid': True
                    })
        
        return poems
    
    def _is_example_copy(self, text: str) -> bool:
        """Check if the generated text is just copying the example"""
        example_phrases = [
            'cherry blossoms fall',
            'soft petals on gentle breeze',
            'golden light descends',
            'painting clouds',
            'day surrenders peace'
        ]
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in example_phrases)
    
    def _count_syllables(self, word: str) -> int:
        """Improved syllable counter"""
        word = word.lower().strip()
        if not word:
            return 0
            
        # Common word exceptions
        exceptions = {
            'the': 1, 'a': 1, 'an': 1, 'of': 1, 'in': 1, 'on': 1,
            'fire': 2, 'hour': 2, 'our': 2, 'flower': 2, 'power': 2,
            'ocean': 2, 'quiet': 2, 'riot': 2, 'science': 2,
            'being': 2, 'going': 2, 'seeing': 2,
            'beautiful': 3, 'different': 3, 'family': 3,
            'evening': 2, 'every': 2, 'several': 3
        }
        
        if word in exceptions:
            return exceptions[word]
        
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for i, char in enumerate(word):
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Adjust for silent 'e' at end
        if word.endswith('e') and len(word) > 2:
            syllable_count -= 1
        
        # Handle -ed endings
        if word.endswith('ed') and len(word) > 3:
            # Check if 'ed' is silent
            if word[-3] not in vowels:
                syllable_count += 1  # Already subtracted for 'e'
        
        # Ensure at least 1 syllable
        return max(1, syllable_count)
    
    def _line_syllables(self, line: str) -> int:
        """Count syllables in a line"""
        # Remove punctuation
        import string
        line = line.translate(str.maketrans('', '', string.punctuation.replace("'", "")))
        words = line.split()
        total = sum(self._count_syllables(word) for word in words)
        return total
    
    def _is_valid_haiku_line(self, line: str, target_syllables: int) -> bool:
        """Check if line matches the target syllable count strictly"""
        syllables = self._line_syllables(line)
        # Strict: must be exactly target syllables
        return syllables == target_syllables
    
    def _validate_haiku_structure(self, haiku: str) -> bool:
        """Check if a haiku follows the 5-7-5 syllable pattern"""
        lines = haiku.strip().split('\n')
        if len(lines) != 3:
            return False
        
        expected = [5, 7, 5]
        for i, line in enumerate(lines):
            syllables = self._line_syllables(line)
            if syllables != expected[i]:
                return False
        return True
    
    def _extract_haiku(self, text: str, prompt: str) -> str:
        """Extract haiku from generated text without strict syllable validation"""
        # Extract content after "Topic: {prompt}"
        marker = f"Topic: {prompt}"
        if marker in text:
            text = text.split(marker)[-1]
        
        # Remove common instruction artifacts
        text = text.replace("Write haiku", "").replace("5-7-5", "")
        text = text.strip()
        
        # Split into lines and filter
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        haiku_lines = []
        
        for line in lines[:15]:  # Check first 15 lines
            # Skip lines that look like instructions
            skip_patterns = [
                'topic:', 'write', 'haiku', 'syllable', 'format', 'poem',
                'i\'m', 'the following'
            ]
            
            line_lower = line.lower()
            
            # Skip if contains instruction patterns
            if any(pattern in line_lower for pattern in skip_patterns):
                continue
            
            # Skip if too long (prose-like) or too short
            if len(line) < 3 or len(line) > 80:
                continue
            
            # Accept the line - no syllable filtering here
            haiku_lines.append(line)
            if len(haiku_lines) == 3:
                break
        
        if len(haiku_lines) >= 3:
            return '\n'.join(haiku_lines[:3])
        
        return ''
    
    def _clean_haiku(self, text: str, prompt: str) -> str:
        """Clean and format the generated text into a haiku"""
        # Debug output removed for cleaner interface
        
        # Extract content after "Topic: {prompt}"
        marker = f"Topic: {prompt}"
        if marker in text:
            text = text.split(marker)[-1]
        
        # Remove common instruction artifacts
        text = text.replace("Write haiku", "").replace("5-7-5", "")
        text = text.strip()
        
        # Split into lines and filter
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        haiku_lines = []
        
        for line in lines[:15]:  # Check more lines
            # Skip lines that look like instructions or prose
            skip_patterns = [
                'topic:', 'write', 'haiku', 'syllable', 'format', 'poem',
                'i\'m', 'the following', 'is a', 'are a', 'can be', 
                'this is', 'it is', 'you can', 'we can'
            ]
            
            line_lower = line.lower()
            
            # Skip if contains instruction patterns
            if any(pattern in line_lower for pattern in skip_patterns):
                continue
            
            # Skip if too long (prose-like) or too short
            if len(line) < 5 or len(line) > 60:
                continue
            
            # Skip if has prose markers (many function words)
            if line_lower.count(' the ') > 2 or line_lower.count(' a ') > 2:
                continue
            
            # Check if line fits haiku syllable pattern
            if len(haiku_lines) == 0:
                # First line (5 syllables) - must reference the prompt topic
                if prompt.lower() not in line.lower() and not self._is_valid_haiku_line(line, 5):
                    continue
            elif len(haiku_lines) == 1:
                if not self._is_valid_haiku_line(line, 7):
                    continue
            elif len(haiku_lines) == 2:
                if not self._is_valid_haiku_line(line, 5):
                    continue
            
            haiku_lines.append(line)
            if len(haiku_lines) == 3:
                break
        
        if len(haiku_lines) >= 3:
            return '\n'.join(haiku_lines[:3])
        
        # Try splitting on punctuation if we have fewer lines
        if len(haiku_lines) in [1, 2]:
            extended_lines = []
            for line in haiku_lines:
                sublines = [s.strip() for s in re.split(r'[,;]\s+', line) if len(s.strip()) > 3]
                extended_lines.extend(sublines)
            
            if len(extended_lines) >= 3:
                return '\n'.join(extended_lines[:3])
        
        return ''
    
    def _fallback_haiku(self) -> str:
        """Return a fallback haiku if generation fails"""
        return "Quiet moon on pond\nsoft wind counts the sleeping reeds\nstarlight keeps its watch"
    
    def generate_poem(
        self,
        prompt: str,
        style: str = 'haiku',
        temperature: float = 0.7,
        num_return_sequences: int = 1,
        max_new_tokens: int = 50,
        **kwargs
    ) -> List[str]:
        """Generate poem with style flexibility (compatible with existing interface)"""
        if style.lower() == 'haiku':
            return self.generate_haiku(prompt, temperature, num_return_sequences, max_new_tokens)
        else:
            # For other styles, generate with appropriate prompt
            if self.model_type == 'seq2seq':
                instruction = f"Write a {style} poem about {prompt}. Be creative and poetic."
            else:
                instruction = f"Write a beautiful {style} about {prompt}:\n\n"
            
            input_ids = self.tokenizer.encode(instruction, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            poems = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            return poems


def load_pretrained_model(model_name: str = 'gpt-neo-125m') -> PretrainedPoetryModel:
    """Convenience function to load a pre-trained model"""
    return PretrainedPoetryModel(model_name)


if __name__ == "__main__":
    # Test with different models
    print("Available models:")
    for name, config in PretrainedPoetryModel.AVAILABLE_MODELS.items():
        print(f"  {name}: {config['description']}")
    
    print("\n" + "="*60)
    print("Testing GPT-Neo-125M")
    print("="*60)
    
    model = load_pretrained_model('gpt-neo-125m')
    haikus = model.generate_haiku('spring cherry blossoms', num_return_sequences=2)
    
    for i, haiku in enumerate(haikus, 1):
        print(f"\nHaiku {i}:")
        print(haiku)
