"""
Load and manage the Kaggle-trained poetry generator model
"""
import torch
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel
import re
from typing import List
from datetime import datetime

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
        print("Loading base GPT-2 (the Body)...")
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
        print("Merged model loaded")

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
        # Normalize prompt: remove any existing tag and build a consistent prompt
        content = prompt.replace('<POETRY>', '').strip()

        if content:
            enhanced_prompt = f"<POETRY> {style} about {content}\n<POEM>\n"
        else:
            enhanced_prompt = f"<POETRY> {style} about\n<POEM>\n"

        input_ids = self.tokenizer.encode(enhanced_prompt, return_tensors='pt').to(self.device)

        # For haiku, use ensemble approach from the start with better sampling
        if style and style.lower() == 'haiku':
            sample_count = max(20, num_return_sequences * 3)
            generation_temperature = 0.6
            generation_top_p = 0.9
        else:
            sample_count = num_return_sequences
            generation_temperature = temperature
            generation_top_p = top_p

        with torch.no_grad():
            # Prepare generation arguments
            if max_new_tokens and max_new_tokens > 0:
                computed_max_length = len(input_ids[0]) + max_new_tokens
            else:
                computed_max_length = max(max_length, len(input_ids[0]) + 1)

            generation_kwargs = {
                'input_ids': input_ids,
                'max_length': computed_max_length,
                'num_return_sequences': sample_count,
                'temperature': generation_temperature,
                'top_p': generation_top_p,
                'do_sample': True,
                'pad_token_id': self.tokenizer.eos_token_id,
                'attention_mask': torch.ones_like(input_ids),
                'repetition_penalty': 1.2,
                'no_repeat_ngram_size': 2,
            }
            
            if min_new_tokens > 0:
                generation_kwargs['min_length'] = len(input_ids[0]) + min_new_tokens

            outputs = self.model.generate(**generation_kwargs)

        poems = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]

        # Clean up the output to remove prompt-like prefixes that may be left
        # after decoding with `skip_special_tokens=True` (which removes the
        # <POETRY> token). We remove optional leading `<POETRY>`, the style
        # label (e.g. "haiku:"), and whitespace.
        import re
        cleaned_poems = []
        for poem in poems:
            p = poem.strip()

            # Remove any leftover tags anywhere in the output
            p_no_tags = re.sub(r'</?POETRY>', '', p, flags=re.IGNORECASE)
            p_no_tags = re.sub(r'</?POEM>', '', p_no_tags, flags=re.IGNORECASE)

            # Remove an echoed leading style/seed header like "haiku about monkey"
            try:
                pattern = rf'^\s*{re.escape(style)}(?:\s+about\s+{re.escape(content)})?\s*[:\-–]?\s*'
                p_no_tags = re.sub(pattern, '', p_no_tags, flags=re.IGNORECASE)
            except Exception:
                # If content contains characters that break the regex, fall back
                p_no_tags = re.sub(rf'^\s*{re.escape(style)}\s*[:\-–]?\s*', '', p_no_tags, flags=re.IGNORECASE)

            # Soft lines: minimal cleaning (keep most content)
            soft_lines = [ln.strip() for ln in p_no_tags.splitlines() if ln.strip()]
            # Clean up double spaces and spacing artifacts within each line
            soft_lines = [re.sub(r'\s+', ' ', ln) for ln in soft_lines]
            soft_lines = [re.sub(r'\s+([.,;!?])', r'\1', ln) for ln in soft_lines]
            
            # For haiku, try to detect if lines were incorrectly merged
            # If we have 1-2 lines but see capital letters mid-text, try to split
            if style and style.lower() == 'haiku' and len(soft_lines) < 3:
                new_lines = []
                for ln in soft_lines:
                    # Split on capital letter preceded by lowercase (likely sentence boundary)
                    segments = re.split(r'(?<=[a-z])\s+(?=[A-Z])', ln)
                    new_lines.extend(segments)
                if 2 <= len(new_lines) <= 4:
                    soft_lines = new_lines

            # Aggressive filter: drop lines that look like instructions
            instr_patterns = [
                r'\bwrite\b',
                r'\bwrite\s+about\b',
                r'\bhas\s+\d+\s+lines\b',
                r'\bsyllable\b',
                r'\b(first|second|third)\b',
                r'\bthis\s+line\b',
                r'\b\w+\s*:\s*write\b',
            ]
            aggressive_lines = [ln for ln in soft_lines if not any(re.search(pat, ln, re.IGNORECASE) for pat in instr_patterns)]

            # Prefer aggressive result if it yields poem content; otherwise fall
            # back to a softer cleaned result that only strips obvious leading
            # instruction headers.
            if aggressive_lines:
                cleaned = '\n'.join(aggressive_lines).strip()
                cleaned_poems.append(cleaned)
                continue

            # Soft fallback: remove only clear header-like first lines
            soft_lines_copy = soft_lines[:]
            if soft_lines_copy:
                first = soft_lines_copy[0]
                if re.search(r'^\s*(write\b|has\s+\d+\s+lines\b|\w+\s*:\s*write\b|haiku\s*:|sonnet\s*:)', first, re.IGNORECASE):
                    soft_lines_copy.pop(0)

            cleaned = '\n'.join(soft_lines_copy).strip()
            cleaned_poems.append(cleaned)

        # Post-generation safety check: detect common unsafe keywords and either
        # attempt a safer regeneration or replace with a neutral fallback.
        def _is_unsafe(text: str) -> bool:
            if not text:
                return False
            # More precise patterns to catch actual unsafe content, not just individual words
            unsafe_patterns = [
                r'\brap(e|ed|ing)\b.*\b(woman|girl|child|boy)\b',  # rape + victim
                r'\b(woman|girl|child|boy)\b.*\brap(e|ed|ing)\b',  # victim + rape
                r'\bsexual(ly)?\s+(assault|violence|abuse)\b',  # sexual violence phrases
                r'\b(kill|murder)(ed|ing)?\s+(child|baby|infant)\b',  # killing children
                r'\bgang\s+rap(e|ed|ing)\b',  # gang rape
                r'\bgraphic\s+violence\b', # graphic violence
                r'\bslaughter(ed|ing)\s+(child|people|woman|men)\b',  # slaughter + victims
                r'\b(gore|blood)\s+(splatt|dripp|pool)\b',  # gore descriptions
            ]
            combined = re.compile('|'.join(unsafe_patterns), flags=re.IGNORECASE)
            return bool(combined.search(text))

        # --- Haiku-specific ensemble & scoring to improve coherence ---
        def _syllables_in_word(word: str) -> int:
            w = word.lower().strip()
            if not w:
                return 0
            # Very small heuristic syllable estimator
            w = re.sub(r'[^a-z]', '', w)
            if not w:
                return 0
            vowels = 'aeiou'
            count = 0
            prev_vowel = False
            for ch in w:
                is_v = ch in vowels
                if is_v and not prev_vowel:
                    count += 1
                prev_vowel = is_v
            # silent e heuristic
            if w.endswith('e') and count > 1:
                count -= 1
            if count == 0:
                count = 1
            return count

        def _syllables_in_line(line: str) -> int:
            return sum(_syllables_in_word(tok) for tok in line.split())

        def _is_coherent(text: str) -> bool:
            """Check if text appears coherent (not just word salad)"""
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if not lines or len(lines) > 5:
                return False
            
            # Check for excessive capitalization mid-sentence
            for line in lines:
                words = line.split()
                if len(words) > 2:
                    # More than half the words capitalized (excluding first) = suspicious
                    caps = sum(1 for w in words[1:] if w and w[0].isupper())
                    if caps > len(words) / 2:
                        return False
            
            # Check for reasonable word count per line
            for line in lines:
                word_count = len(line.split())
                if word_count < 2 or word_count > 15:
                    return False
            
            return True

        def _haiku_score(text: str) -> int:
            # Lower is better. Penalize deviation from 5-7-5 and non-3-line structure.
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if len(lines) != 3:
                # heavy penalty for wrong line count
                return 1000 + abs(len(lines) - 3) * 50
            
            # Penalize lines that are too short or too long in characters
            for line in lines:
                if len(line) < 3:  # too short
                    return 1000
                if len(line) > 60:  # too long for a haiku line
                    return 500
            
            s1 = _syllables_in_line(lines[0])
            s2 = _syllables_in_line(lines[1])
            s3 = _syllables_in_line(lines[2])
            
            # Syllable deviation score
            score = abs(s1 - 5) + abs(s2 - 7) + abs(s3 - 5)
            
            # Penalize lines with 0 syllables (shouldn't happen but just in case)
            if s1 == 0 or s2 == 0 or s3 == 0:
                score += 500
            
            return score

        # If haiku, score and select best candidates
        if style and style.lower() == 'haiku':
            scored_candidates = []
            for c in cleaned_poems:
                if c and len(c.strip()) > 5 and _is_coherent(c):  # basic sanity check
                    score = _haiku_score(c)
                    scored_candidates.append((score, c))
            
            # Sort by score (lower is better) and take top candidates
            scored_candidates.sort(key=lambda x: x[0])
            # Take more than needed for safety filtering
            if scored_candidates:
                cleaned_poems = [c for (score, c) in scored_candidates[:max(10, num_return_sequences * 2)]]
            # If no coherent candidates, keep originals and hope safety filter helps
            # (it will likely trigger fallback)

        # Safety check with retries
        safe_poems: List[str] = []
        for p in cleaned_poems:
            if _is_unsafe(p):
                # Log the unsafe detection with timestamp, prompt and original text
                try:
                    with open('unsafe_generation.log', 'a', encoding='utf-8') as lf:
                        lf.write(f"--- {datetime.utcnow().isoformat()}Z ---\n")
                        lf.write(f"PROMPT: {enhanced_prompt}\n")
                        lf.write("ORIGINAL_OUTPUT:\n")
                        lf.write(p + "\n\n")
                except Exception:
                    pass

                chosen = None
                seen_texts = {p.strip()}

                # Try up to 3 retries with stricter sampling parameters to steer away
                # from unsafe topics. Each retry makes sampling more conservative.
                for attempt, (t_factor, p_factor) in enumerate([(0.5, 0.5), (0.35, 0.3), (0.2, 0.2)], start=1):
                    retry_kwargs = generation_kwargs.copy()
                    retry_kwargs.update({
                        'temperature': min(float(temperature) * t_factor, float(temperature)),
                        'top_p': min(float(top_p) * p_factor, float(top_p)),
                        'do_sample': True,
                        'num_return_sequences': 1,
                    })

                    try:
                        with torch.no_grad():
                            retry_outputs = self.model.generate(**retry_kwargs)
                        retry_text = self.tokenizer.decode(retry_outputs[0], skip_special_tokens=True)
                    except Exception:
                        retry_text = ''

                    # Minimal cleaning
                    retry_clean = re.sub(r'</?POETRY>', '', retry_text, flags=re.IGNORECASE)
                    retry_clean = re.sub(r'</?POEM>', '', retry_clean, flags=re.IGNORECASE).strip()

                    # Record retry attempt
                    try:
                        with open('unsafe_generation.log', 'a', encoding='utf-8') as lf:
                            lf.write(f"RETRY {attempt} (temp={retry_kwargs.get('temperature')}, top_p={retry_kwargs.get('top_p')}):\n")
                            lf.write(retry_clean + "\n\n")
                    except Exception:
                        pass

                    if retry_clean and not _is_unsafe(retry_clean) and retry_clean.strip() not in seen_texts:
                        chosen = retry_clean
                        break
                    if retry_clean:
                        seen_texts.add(retry_clean.strip())

                if chosen:
                    safe_poems.append(chosen)
                else:
                    # Fallback neutral haiku if all retries fail to remove unsafe content
                    fallback = (
                        "Quiet moon on pond\nsoft wind counts the sleeping reeds\nstarlight keeps its watch"
                    )
                    try:
                        with open('unsafe_generation.log', 'a', encoding='utf-8') as lf:
                            lf.write(f"FINAL_FALLBACK:\n{fallback}\n\n")
                    except Exception:
                        pass
                    safe_poems.append(fallback)
            else:
                safe_poems.append(p)

        # Return the requested number of poems
        return safe_poems[:num_return_sequences] if len(safe_poems) > num_return_sequences else safe_poems

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