import requests
import json
import os
import re

def refine_with_api(raw_poetry, style="haiku", seed=""):
    """
    Refines raw GPT-2 text into structured poetry using an API.
    Forces all output to lowercase and scrubs technical noise.
    """

    # 1. PRE-CLEANING: Scrub technical noise
    raw_poetry = re.sub(r'(rawdownload|embedreport|print|http\S+|ãâ|â|ã|\[.*?\]|\{.*?\})', '', raw_poetry)
    raw_poetry = raw_poetry.encode("ascii", "ignore").decode().strip()

    # Check if API key is provided via environment variable
    api_key = os.getenv("POETRON_API_KEY")
    
    # If no API key is set via environment variable, return the raw poem without refinement
    if not api_key:
        print("[WARNING] API key not set via environment variable. Skipping API refinement.")
        return raw_poetry

    api_url = "https://ai.hackclub.com/proxy/v1/chat/completions"

    # 2. STYLE CONTROLS
    controls = {
        "haiku": {
            "instructions": "exactly 3 lines. syllable count: 5-7-5.",
            "constraints": "use nature imagery. avoid punctuation where possible."
        },
        "sonnet": {
            "instructions": "14 lines. traditional rhyme scheme.",
            "constraints": "maintain iambic rhythm."
        },
        "freeverse": {
            "instructions": "expressive imagery. no strict meter.",
            "constraints": "focus on emotional depth and imagery."
        }
    }

    # 3. BUILD PROMPT with a small few-shot examples to reduce gibberish
    control = controls.get(style.lower(), controls["freeverse"])

    few_shot_examples = {
        'haiku': (
            "Raw text: golden horizon, waves and light.\nTheme: ocean\n\nRefined:\n<<<POEM>>>\nGolden tide at dawn\nLight slips across the water\nSoft breath of the sea\n<<<END>>>"
        ),
        'sonnet': (
            "Raw text: love and time entwined in quiet mornings.\nTheme: love\n\nRefined:\n<<<POEM>>>\nWhen morning draws its curtains on the day,\nSoft light recalls the warmth of yesterday;\n<<<END>>>"
        ),
        'freeverse': (
            "Raw text: wandering city streets, neon, rain.\nTheme: night\n\nRefined:\n<<<POEM>>>\nNeon pools on pavement, footsteps melt into rain,\nA hush that knows the city by its breath\n<<<END>>>"
        )
    }

    example_block = few_shot_examples.get(style.lower(), '')

    prompt_lines = []
    if example_block:
        prompt_lines.append("Examples:\n")
        prompt_lines.append(example_block)
        prompt_lines.append('\n---\n')

    prompt_lines.append(f"Instructions: {control['instructions']} {control['constraints']}")
    prompt_lines.append(f"Raw text: {raw_poetry}")
    prompt_lines.append(f"Theme: {seed}")
    prompt_lines.append("Refined poem between <<<POEM>>> and <<<END>>>:")

    prompt = "\n".join(prompt_lines).strip()

    # 4. API REQUEST
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openai/gpt-5-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful poetry editor. Return only the refined poem between <<<POEM>>> and <<<END>>> with no extra commentary."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.6,
        "max_tokens": 200
    }

    try:
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()

        result = response.json()

        def _extract_poem(text: str) -> str:
            if not text:
                return ''
            text = text.strip()
            if '<<<POEM>>>' in text and '<<<END>>>' in text:
                try:
                    return text.split('<<<POEM>>>', 1)[1].split('<<<END>>>', 1)[0].strip()
                except Exception:
                    return text
            return text

        def _is_valid(poem_text: str, style_name: str) -> bool:
            if not poem_text:
                return False
            # require a minimum number of alphabetic characters
            letters_only = re.sub(r'[^A-Za-z]', '', poem_text)
            if len(letters_only) < 20:
                return False
            # require at least one newline (multi-line poem)
            lines = [l for l in poem_text.splitlines() if l.strip()]
            if len(lines) == 0:
                return False
            if style_name.lower() == 'haiku' and len(lines) != 3:
                return False
            if style_name.lower() == 'sonnet' and len(lines) < 10:
                return False
            return True

        if 'choices' in result and len(result['choices']) > 0:
            choice = result['choices'][0]
            content = None
            if isinstance(choice.get('message'), dict):
                content = choice['message'].get('content')
            content = content or choice.get('text') or ''

            poem = _extract_poem(content)
            if _is_valid(poem, style):
                return poem

            # Retry once with stricter system message if output invalid
            try:
                retry_payload = {
                    "model": payload.get('model'),
                    "messages": [
                        {"role": "system", "content": "You are a strict poetry editor. Output ONLY the poem between <<<POEM>>> and <<<END>>> with no commentary."},
                        {"role": "user", "content": prompt + "\n\nIf your previous output was invalid, now output only the poem between <<<POEM>>> and <<<END>>>."}
                    ],
                    "temperature": 0.45,
                    "max_tokens": payload.get('max_tokens', 200)
                }

                retry_resp = requests.post(api_url, json=retry_payload, headers=headers, timeout=15)
                retry_resp.raise_for_status()
                retry_result = retry_resp.json()
                if 'choices' in retry_result and len(retry_result['choices']) > 0:
                    retry_choice = retry_result['choices'][0]
                    retry_content = None
                    if isinstance(retry_choice.get('message'), dict):
                        retry_content = retry_choice['message'].get('content')
                    retry_content = retry_content or retry_choice.get('text') or ''
                    retry_poem = _extract_poem(retry_content)
                    if _is_valid(retry_poem, style):
                        return retry_poem

            except Exception:
                pass

        print("[ERROR] API response format unexpected or invalid, returning original")
        return raw_poetry

    except Exception as e:
        print(f"[ERROR] API refinement failed: {str(e)}, returning original")
        return raw_poetry