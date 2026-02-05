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
        print("⚠️  API key not set via environment variable. Skipping API refinement.")
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
        
        if 'choices' in result and len(result['choices']) > 0:
            refined = result['choices'][0]['message']['content'].strip()
            return refined
        else:
            print("❌ API response format unexpected, returning original")
            return raw_poetry
            
    except Exception as e:
        print(f"❌ API refinement failed: {str(e)}, returning original")
        return raw_poetry