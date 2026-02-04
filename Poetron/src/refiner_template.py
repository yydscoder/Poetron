import requests
import json
import os
import re

def refine_with_api(raw_poetry, style="haiku", seed=""):
    """
    Refines raw GPT-2 text into structured poetry using an API.
    Forces all output to lowercase and scrubs technical noise.
    """
    
    # 1. PRE-CLEANING: Scrub noise
    raw_poetry = re.sub(r'(rawdownload|embedreport|print|http\S+|ãâ|â|ã|\[.*?\]|\{.*?\})', '', raw_poetry)
    raw_poetry = raw_poetry.encode("ascii", "ignore").decode().strip()

    api_url = "https://ai.hackclub.com/proxy/v1/chat/completions"
    api_key = os.getenv("POETRON_API_KEY")

    # If no API key is set via environment variable, return the raw poem without refinement
    if not api_key:
        print("API key not set via environment variable. Skipping API refinement.")
        return raw_poetry 

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
            "constraints": "amplify the emotional mood."
        }
    }

    style_key = style.lower() if style.lower() in controls else "freeverse"
    current_style = controls[style_key]

    # 3. THE SYSTEM PROTOCOL (Now requesting lowercase)
    system_message = f"""you are a poetry transformation engine. 
    your output must be a polished {style_key.upper()}.

    strict rules:
    - use ONLY lowercase letters. no capital letters at the start of lines.
    - output ONLY the poem. no preamble.
    - delete technical jargon like 'download' or 'null'.
    - {current_style['instructions']}
    - {current_style['constraints']}"""

    user_message = f"raw data: {raw_poetry}\ntheme: {seed}\ntransformed poem:"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini", 
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.5, 
        "max_tokens": 300
    }

    try:
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

        if 'choices' in result:
            refined_poem = result['choices'][0]['message']['content'].strip()
            
            # FINAL ENFORCEMENT: Force everything to lowercase via Python
            # This is a safety net in case the AI ignores the system prompt
            refined_poem = refined_poem.lower()
            
            # Remove any labels the AI might have accidentally included
            refined_poem = re.sub(r'^(poem|haiku|refined|result):', '', refined_poem)
            
            return refined_poem.strip().strip('"')
        
        return raw_poetry.lower()
            
    except Exception as e:
        print(f"api error: {e}")
        return raw_poetry.lower()

if __name__ == "__main__":
    # Example Test
    test_input = "Trees like to be blown up. Leaves falling out of the sky."
    print(refine_with_api(test_input, "haiku", "trees"))