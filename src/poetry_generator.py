"""
Poetry generation functionality for the Poetry Generator
"""

import torch
import random
from pathlib import Path
import re
import os

try:
    from .utils import format_poem_for_style
    from .refiner import refine_with_api
except ImportError:
    from utils import format_poem_for_style
    from refiner import refine_with_api


def generate_poem(
    style: str,
    seed: str = "",
    length: int = 50,
    model_path: str = "models/poetry_model",
    temperature: float = 0.8,
    max_new_tokens: int = 100
):
    """
    Generate a poem in the specified style using LoRA model.
    Falls back to rule-based generation if model unavailable.

    Args:
        style (str): The style of poem to generate ('haiku', 'sonnet', 'freeverse')
        seed (str): Seed words or themes for the poem
        length (int): Desired length of the poem (for free verse)
        model_path (str): Path to the trained LoRA adapter
        temperature (float): Sampling temperature for generation
        max_new_tokens (int): Maximum number of new tokens to generate

    Returns:
        str: Generated poem
    """
    # Prepare the prompt with style token and more specific instructions
    style_token = f"<POETRY>"
    
    # Enhance the prompt with style-specific instructions
    if style.lower() == 'haiku':
        style_instruction = f"Write a traditional haiku about {seed or 'nature'}. A haiku has 3 lines with a 5-7-5 syllable pattern."
    elif style.lower() == 'sonnet':
        style_instruction = f"Write a sonnet about {seed or 'love'}. A sonnet has 14 lines and usually follows a specific rhyme scheme."
    elif style.lower() == 'freeverse':
        style_instruction = f"Write a free verse poem about {seed or 'life'}. Free verse has no set meter or rhyme scheme."
    else:
        style_instruction = f"Write a poem in {style} style about {seed or 'life'}."

    prompt = f"{style_token} {style_instruction}".strip()

    try:
        # Try to load and use the LoRA model (prefer package-relative import)
        try:
            from .load_kaggle_model import load_kaggle_model
        except Exception:
            from load_kaggle_model import load_kaggle_model

        model_path_obj = Path(model_path)

        # If the provided path exists but does not directly contain a model
        # indicator file (adapter_config.json or config.json), attempt to
        # locate a nested model directory (common with Kaggle artifacts).
        if model_path_obj.exists():
            # If the directory doesn't look like a model, search for nested model
            if not (model_path_obj / "adapter_config.json").exists() and not (model_path_obj / "config.json").exists():
                # Search recursively for known model indicator files
                nested_adapter = next(model_path_obj.rglob('adapter_config.json'), None)
                nested_config = next(model_path_obj.rglob('config.json'), None)
                candidate = None
                if nested_adapter:
                    candidate = nested_adapter.parent
                elif nested_config:
                    candidate = nested_config.parent

                if candidate:
                    print(f"[INFO] Found nested model at {candidate}, using that path instead of {model_path}")
                    model_path_obj = candidate

            print(f"[INFO] Loading trained LoRA model from {model_path_obj}...")

            # Load model (Body + Brain)
            poetry_model = load_kaggle_model(model_path)
            poetry_model.load_tokenizer()

            # Generate using the model
            # Haiku: generate multiple candidates and pick the best-scored one
            if style.lower() == 'haiku':
                candidates = []
                tries = 30
                gen_temperature = min(0.45, float(temperature))
                gen_max_new_tokens = min(40, int(max_new_tokens))
                for _ in range(tries):
                    out = poetry_model.generate_poem(
                        prompt=seed,
                        max_length=gen_max_new_tokens,
                        temperature=gen_temperature,
                        num_return_sequences=1,
                        style=style,
                        max_new_tokens=gen_max_new_tokens
                    )
                    if out and out[0]:
                        cleaned_out = _clean_candidate(out[0])
                        if cleaned_out:
                            candidates.append(cleaned_out)

                # Postprocess candidates and score
                best = None
                best_score = float('inf')
                for cand in candidates:
                    cleaned = cand.strip()
                    formatted = format_poem_for_style(cleaned, 'haiku')
                    haiku = enforce_haiku_structure(formatted)
                    lines = [ln.strip() for ln in haiku.split('\n')]
                    score = _score_haiku(lines, seed)
                    if score < best_score:
                        best_score = score
                        best = haiku

                # Apply repetition filter and a final post-clean + aggressive
                # cleanup to the selected best
                if best:
                    try:
                        best = _remove_adjacent_repetition(best)
                        best = post_clean_poem(best)
                        best = _aggressive_cleanup(best, seed)
                    except Exception:
                        pass

                poems = [best] if best else []
            else:
                poems = poetry_model.generate_poem(
                    prompt=seed,
                    max_length=max_new_tokens,
                    temperature=temperature,
                    num_return_sequences=1,
                    style=style,  # Pass the style to influence generation parameters
                    max_new_tokens=max_new_tokens
                )

            generated_text = poems[0] if poems else generate_fallback_poem(style, seed)

            # The model wrapper returns cleaned text (prompt prefixes stripped).
            # Use the returned string directly rather than attempting to re-strip
            # the original prompt here to avoid accidental truncation.
            raw_poem = generated_text.strip()

            # If the cleaned model output is empty (aggressive filtering may
            # have removed content), fall back to rule-based generator so the
            # user still receives a poem instead of an empty string.
            if not raw_poem:
                print("[WARN] Model output empty after cleaning; using fallback generator")
                return generate_fallback_poem(style, seed)

            # Clean up the raw poem to remove special characters and control characters
            import re
            # Remove control characters (except newlines and tabs)
            cleaned_raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', raw_poem)
            # Remove extra whitespace and normalize line breaks
            cleaned_raw = re.sub(r'\r\n', '\n', cleaned_raw)
            cleaned_raw = re.sub(r'\r', '\n', cleaned_raw)
            cleaned_raw = re.sub(r'\n+', '\n', cleaned_raw)

            # Remove the "seed style:" part that appears in the output
            # This pattern appears to be "trees haiku:" or similar
            cleaned_raw = re.sub(r'\w+\s+' + style + r':?\s*', '', cleaned_raw, flags=re.IGNORECASE)

            cleaned_raw = cleaned_raw.strip()

            # Apply style formatting
            formatted_raw = format_poem_for_style(cleaned_raw, style)
            # Post-clean the formatted poem to remove artifacts.
            try:
                formatted_raw = post_clean_poem(formatted_raw)
            except Exception:
                pass
            # For haiku, enforce 3-line structure for better coherence
            if style.lower() == 'haiku':
                try:
                    # First enforce 3-line structure, then attempt strict 5-7-5
                    formatted_raw = enforce_haiku_structure(formatted_raw)
                    formatted_raw = enforce_exact_haiku(formatted_raw, seed)
                except Exception:
                    pass
            # Ensure haiku contains the seed word/phrase. If not present,
            # inject the seed into the first line to guarantee the poem is
            # about the requested subject.
            if style.lower() == 'haiku' and seed:
                try:
                    if seed.lower() not in formatted_raw.lower():
                        # Ensure three lines, then prepend the seed to line 1
                        haiku_lines = enforce_haiku_structure(formatted_raw).split('\n')
                        if not haiku_lines[0].lower().startswith(seed.lower()):
                            haiku_lines[0] = f"{seed} {haiku_lines[0]}".strip()
                        formatted_raw = '\n'.join(haiku_lines[:3])
                except Exception:
                    pass
            # Conditionally refine via API if POETRON_API_KEY is provided
            api_key = os.getenv('POETRON_API_KEY')
            if api_key:
                try:
                    refined_poem = refine_with_api(formatted_raw, style, seed)
                    # If refinement returned an error string, fall back to formatted_raw
                    if isinstance(refined_poem, str) and refined_poem.startswith("Error:"):
                        return formatted_raw
                    # Post-clean the refined poem as well
                    try:
                        refined_poem = post_clean_poem(refined_poem)
                    except Exception:
                        pass
                    return refined_poem
                except Exception:
                    return formatted_raw

            return formatted_raw
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")

    except Exception as e:
        # Fallback to rule-based generation
        print(f"Using fallback poem generator ({str(e)[:50]}...)")
        return generate_fallback_poem(style, seed)


def generate_fallback_poem(style: str, seed: str = ""):
    """
    Generate a fallback poem if the model fails.

    Args:
        style (str): The style of poem to generate
        seed (str): Seed words or themes for the poem

    Returns:
        str: Fallback poem for flagging
    """
    if style.lower() == 'haiku':
        lines = [
            seed if seed else "Silent morning dew",
            "Whispers to the awakening earth",
            "Sunlight paints the leaves"
        ]
        return "\n".join(lines[:3])

    elif style.lower() == 'sonnet':
        lines = [
            f"When {seed} fills the air with gentle thought," if seed else "When morning light breaks through the silent dawn,",
            "And birds begin their chorus sweet and clear,",
            "The world awakens from night's peaceful sleep,",
            "As flowers bloom beneath the sky so blue.",
            "",
            "The gentle breeze carries a whispered prayer,",
            "Of love that grows like vines along the wall,",
            "While memories dance in sunlit summer air,",
            "And time moves slow, yet swift as eagles fly.",
            "",
            "So moments pass like clouds across the sky,",
            "Yet linger in the heart's most sacred space,",
            "Where dreams take flight and never say goodbye,",
            "To beauty found in time's eternal grace."
        ]
        return "\n".join(lines[:14])

    else:  # freeverse
        lines = [
            seed if seed else "The world breathes",
            "In rhythms unknown",
            "To those who rush",
            "Through life's maze",
            "",
            "But pause",
            "And listen",
            "To the silence",
            "Between heartbeats",
            "",
            "There lies",
            "The poetry",
            "Of existence"
        ]
        return "\n".join(lines)


def enforce_haiku_structure(text: str) -> str:
    """
    Attempt to format text as a haiku (3 lines with 5-7-5 syllable pattern).

    Note: This is a simplified approximation and doesn't perfectly count syllables.
    A more sophisticated approach would be needed for accurate syllable counting.

    Args:
        text (str): Input text to format as haiku

    Returns:
        str: Text formatted as haiku
    """
    lines = text.split('\n')
    haiku_lines = []

    # Take first three lines or create three lines from the text
    for i in range(3):
        if i < len(lines) and lines[i].strip():
            haiku_lines.append(lines[i].strip())
        else:
            # Create a line from remaining text
            words = ' '.join(lines[i:]).split() if i < len(lines) else text.split()
            if words:
                # Approximate 5-7-5 syllable pattern with word counts (5-7-5 words ≈ 5-7-5 syllables)
                word_counts = [5, 7, 5]
                line_words = words[:word_counts[i]] if i < len(word_counts) else words[:5]
                haiku_lines.append(' '.join(line_words))
                # Remove used words
                words = words[word_counts[i]:] if i < len(word_counts) else words[5:]

    return '\n'.join(haiku_lines[:3])


def _syllables_in_word(word: str) -> int:
    """
    Very small heuristic syllable counter: counts vowel groups, adjusts for
    common silent-e and short words. Not perfect but useful for ranking.
    """
    w = word.lower()
    import re
    w = re.sub(r'[^a-z]', '', w)
    if not w:
        return 0
    # Count vowel groups
    groups = re.findall(r'[aeiouy]+', w)
    count = len(groups)
    # Subtract silent 'e' endings
    if w.endswith('e') and not w.endswith(('le', 'ue')) and count > 1:
        count -= 1
    # Minimum 1
    return max(1, count)


def _score_haiku(lines: list[str], seed: str) -> float:
    """
    Score a 3-line haiku candidate. Lower is better.
    Components:
    - syllable distance from 5-7-5 (primary)
    - whether seed appears (bonus)
    - penalty for non-alpha tokens or very short lines
    """
    target = [5, 7, 5]
    import re
    total_dist = 0
    penalty = 0
    for i, line in enumerate(lines[:3]):
        words = [w for w in re.findall(r"\w+'?\w*", line)]
        syl = sum(_syllables_in_word(w) for w in words) if words else 0
        total_dist += abs(syl - target[i])
        # penalize if line too short
        if len(words) < 2:
            penalty += 1
        # penalize weird tokens
        for w in words:
            if re.search(r'[^A-Za-z\']', w):
                penalty += 0.5

    seed_bonus = 0 if seed and seed.lower() in ' '.join(lines).lower() else 2
    return total_dist + penalty + seed_bonus


def _clean_candidate(text: str) -> str:
    """Lightweight cleaning of model candidates to remove common artifacts."""
    import re
    if not text:
        return ""
    # Remove non-ASCII junk and control chars
    t = re.sub(r'[^	\n\x20-\x7E]+', '', text)
    # Normalize whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    # Remove some known junk tokens that appear in adapter outputs
    t = re.sub(r'\bembedreportprint\b', '', t, flags=re.IGNORECASE)
    t = re.sub(r'\bpione\b', '', t, flags=re.IGNORECASE)
    # Drop words that are purely numeric or single punctuation
    words = [w for w in t.split() if not re.fullmatch(r"[0-9]+", w) and not re.fullmatch(r"\W+", w)]
    # Remove very short tokens (single-letter) except 'I' and 'a'
    words = [w for w in words if len(w) > 1 or w.lower() in ('i', 'a')]
    return ' '.join(words).strip()


def _remove_adjacent_repetition(text: str) -> str:
    """Remove simple adjacent repeated phrases across lines or within a line.

    Examples handled:
    - "the cat the cat" -> "the cat"
    - repeated adjacent lines are collapsed
    """
    import re
    if not text:
        return text
    # Collapse repeated adjacent lines
    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
    new_lines = []
    for ln in lines:
        if new_lines and ln.lower() == new_lines[-1].lower():
            continue
        new_lines.append(ln)

    # Within each line, remove immediate repeated n-grams (handled by collapse),
    # also collapse exact repeated words
    cleaned = []
    for ln in new_lines:
        # remove repeated words like "the the"
        ln = re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', ln, flags=re.IGNORECASE)
        cleaned.append(ln)

    return '\n'.join(cleaned)


def enforce_exact_haiku(text: str, seed: str = "") -> str:
    """
    Attempt to produce a strict 5-7-5 haiku from `text` using a greedy
    syllable allocation algorithm. This is heuristic and will try to balance
    words to match the syllable targets. If it fails to produce good lines,
    falls back to `enforce_haiku_structure` behavior.
    """
    import re

    # Extract words (keep simple contractions)
    words = re.findall(r"\w+'?\w*", text)
    if not words:
        return enforce_haiku_structure(text)

    targets = [5, 7, 5]
    lines = []
    idx = 0

    try:
        for tgt in targets:
            curr = []
            curr_syl = 0
            # Greedily add words until we reach or slightly exceed the target
            while idx < len(words) and curr_syl < tgt:
                w = words[idx]
                s = _syllables_in_word(w)
                curr.append(w)
                curr_syl += s
                idx += 1

            # If we overshot by a lot and there is a next word, try moving the
            # last word to the next line to reduce overshoot.
            if curr and curr_syl - tgt > 2 and idx < len(words):
                last = curr.pop()
                idx -= 1

            lines.append(' '.join(curr).strip())

        # If we didn't consume any words for later lines, try to rebalance by
        # splitting longer lines (soft fallback)
        if any(not ln for ln in lines):
            # Use the simpler structure enforcer if greedy fails
            return enforce_haiku_structure(text)

        # If there are leftover words, append them to the last line
        if idx < len(words):
            extra = ' '.join(words[idx:])
            if lines[-1]:
                lines[-1] = f"{lines[-1]} {extra}"
            else:
                lines[-1] = extra

        # Final cleanup: ensure exactly 3 lines
        final = [ln.strip() for ln in lines][:3]

        # Ensure seed presence
        if seed and seed.lower() not in ' '.join(final).lower():
            final[0] = f"{seed} {final[0]}".strip()

        return '\n'.join(final)
    except Exception:
        return enforce_haiku_structure(text)


def _collapse_repeated_ngrams(line: str, max_n: int = 4) -> str:
    """Remove immediate repeated n-grams in a line (greedy)."""
    import re
    tokens = line.split()
    if not tokens:
        return line
    i = 0
    out = []
    while i < len(tokens):
        removed = False
        # try larger n first
        for n in range(min(max_n, (len(tokens)-i)//2), 0, -1):
            if tokens[i:i+n] == tokens[i+n:i+2*n]:
                # skip the repeated n-gram
                out.extend(tokens[i:i+n])
                i += 2*n
                removed = True
                break
        if not removed:
            out.append(tokens[i])
            i += 1
    return ' '.join(out)


def _fix_contractions(line: str) -> str:
    import re
    s = line
    # common fixes
    s = re.sub(r"\bdont\b", "don't", s, flags=re.IGNORECASE)
    s = re.sub(r"don'\s*know", "don't know", s, flags=re.IGNORECASE)
    s = re.sub(r"\bit'\s*(\w)", r"it's \1", s, flags=re.IGNORECASE)
    s = re.sub(r"\bit'\b", "it's", s, flags=re.IGNORECASE)
    s = re.sub(r"\bi\s*'\s*m\b", "I'm", s, flags=re.IGNORECASE)
    s = re.sub(r"\bI\s'\s*m\b", "I'm", s)
    s = re.sub(r"\bcan'\s*t\b", "can't", s, flags=re.IGNORECASE)
    s = re.sub(r"\bwon'\s*t\b", "won't", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(\w+)'\s+s\b", r"\1's", s)  # e.g., it' s -> it's
    # remove stray spaces before punctuation
    s = re.sub(r"\s+([,.!?;:])", r"\1", s)
    # normalize spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def post_clean_poem(text: str) -> str:
    """Apply lightweight post-cleaning to poem text: contractions, repeated phrases."""
    import re
    if not text:
        return text
    lines = [ln.strip() for ln in text.split('\n')]
    cleaned_lines = []
    for ln in lines:
        if not ln:
            cleaned_lines.append(ln)
            continue
        ln = _clean_candidate(ln)
        ln = _collapse_repeated_ngrams(ln, max_n=3)
        ln = _fix_contractions(ln)
        # capitalize first character of the line
        if ln:
            ln = ln[0].upper() + ln[1:]
        cleaned_lines.append(ln)
    # Remove leading/trailing blank lines
    while cleaned_lines and not cleaned_lines[0].strip():
        cleaned_lines.pop(0)
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()

    return '\n'.join(cleaned_lines)


def _aggressive_cleanup(text: str, seed: str = "") -> str:
    """Perform additional aggressive cleanup to remove trailing fragments,
    single-word stray lines, and normalize a few common patterns.

    This complements `post_clean_poem` and is applied as a final pass.
    """
    import re
    if not text:
        return text

    lines = [ln.strip() for ln in text.split('\n') if ln is not None]
    cleaned = []

    for i, ln in enumerate(lines):
        if not ln:
            continue

        # Remove lines that are a single very short token (likely fragment),
        # unless it matches the seed or is meaningful ('I', 'A').
        tokens = ln.split()
        if len(tokens) == 1:
            tok = tokens[0]
            if len(tok) <= 2 and tok.lower() not in (seed.lower(), 'i', 'a'):
                continue

        # Normalize leading 'am ' to 'I am ' when it appears at line start
        if re.match(r'^\s*am\b', ln, flags=re.IGNORECASE):
            ln = re.sub(r'^\s*am\b', "I am", ln, flags=re.IGNORECASE)

        # Remove trailing single-word fragments like a lone 'The' at the end
        if i == len(lines) - 1:
            if re.fullmatch(r'[Tt]he|[Aa]|[Ii]|\W+', ln):
                continue

        # Remove lines that end with an incomplete punctuation-heavy fragment
        if re.search(r'[\W]$', ln) and not re.search(r'[a-zA-Z]$', ln):
            ln = re.sub(r'[^a-zA-Z\s]+$', '', ln).strip()
            if not ln:
                continue

        cleaned.append(ln)

    # If we ended up empty, return original text (fallback)
    if not cleaned:
        return text

    # Re-join and normalize spacing and punctuation
    out = '\n'.join(cleaned)
    out = re.sub(r"\s+([,.;:!\?])", r"\1", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _lm_score(text: str, poetry_model) -> float:
    """Compute average negative log-likelihood of `text` under the local model.

    Lower is better. Returns a large number on failure.
    """
    import torch
    try:
        if not hasattr(poetry_model, 'tokenizer') or not hasattr(poetry_model, 'model'):
            return float('inf')

        tokenizer = poetry_model.tokenizer
        model = poetry_model.model
        model.eval()

        # Encode and move to model device
        input_ids = tokenizer.encode(text, return_tensors='pt')
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            # outputs.loss present when labels provided
            loss = None
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss.item()
            else:
                # Fallback: compute cross-entropy from logits
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).item()

        # Return average token loss
        return float(loss)
    except Exception:
        return float('inf')


def generate_with_style_control(style: str, seed: str, model_path: str, temperature: float = 0.8):
    """
    Generate a poem with enhanced style control.

    Args:
        style (str): The style of poem to generate
        seed (str): Seed words or themes for the poem
        model_path (str): Path to the trained model
        temperature (float): Sampling temperature

    Returns:
        str: Generated poem with style control
    """
    # This would implement more sophisticated style control
    poem = generate_poem(style, seed, model_path=model_path, temperature=temperature)

    if style.lower() == 'haiku':
        return enforce_haiku_structure(poem)

    return poem