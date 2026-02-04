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

    # 3. BUILD PROMPT
    control = controls.get(style.lower(), controls["freeverse"])
    prompt = f"""
    Transform this raw text into a {style}:
    {control['instructions']}
    {control['constraints']}
    
    Raw text: {raw_poetry}
    
    Theme: {seed}
    
    Refined poem:
    """

    # 4. API REQUEST
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openai/gpt-5-mini",
        "messages": [
            {"role": "system", "content": "You are a poetry editor. Transform raw text into structured poetry."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
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