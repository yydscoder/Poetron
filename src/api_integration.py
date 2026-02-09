"""
API integration for remote poem generation
"""

import requests
import os
from typing import Optional


def generate_poem_via_api(style: str, seed: str, length: int = 50, api_url: Optional[str] = None, api_token: Optional[str] = None, api_model: str = "openai/gpt-5-mini") -> str:
    """
    Generate a poem using a remote API.
    
    Args:
        style (str): The style of poem to generate ('haiku', 'sonnet', 'freeverse')
        seed (str): Seed words or themes for the poem
        length (int): Desired length of the poem
        api_url (str, optional): Custom API URL
        api_token (str, optional): API token for authentication
        
    Returns:
        str: Generated poem
    """
    # Use environment variable for API URL if not provided (default to Hack Club proxy)
    if not api_url:
        api_url = os.getenv('POETRON_API_URL', 'https://ai.hackclub.com/proxy/v1/chat/completions')

    # Use environment variable for API token if not provided
    if not api_token:
        api_token = os.getenv('POETRON_API_KEY')

    if not api_token:
        return "Error: No API token provided. Please set POETRON_API_KEY environment variable."
    
    # Construct the payload for the chat-style API request (HackClub proxy / OpenAI-compatible)
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }

    # Build a concise user prompt
    style_prompts = {
        'haiku': f"Create a haiku about {seed or 'nature'}. Reply only with the poem between <<<POEM>>> and <<<END>>> (3 lines, 5-7-5).",
        'sonnet': f"Create a sonnet about {seed or 'love'}. Reply only with the poem between <<<POEM>>> and <<<END>>> (14 lines).",
        'freeverse': f"Create a free verse poem about {seed or 'life'}. Reply only with the poem between <<<POEM>>> and <<<END>>>."
    }

    user_prompt = style_prompts.get(style.lower(), f"Create a {style} poem about {seed}. Reply only with the poem between <<<POEM>>> and <<<END>>>.")

    payload = {
        "model": api_model,
        "messages": [
            {"role": "system", "content": "You are a poetry editor. Transform prompts into clean, well-formed poems."},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.6,
        "max_tokens": max(64, length * 4)
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()

        # Try chat completion response format
        if isinstance(result, dict) and 'choices' in result and len(result['choices']) > 0:
            choice = result['choices'][0]
            # Chat-style
            content = None
            if 'message' in choice and isinstance(choice['message'], dict):
                content = choice['message'].get('content')
            else:
                content = choice.get('text') or choice.get('message')

            if content:
                text = content.strip()
                # Extract between markers if present
                if '<<<POEM>>>' in text and '<<<END>>>' in text:
                    poem = text.split('<<<POEM>>>', 1)[1].split('<<<END>>>', 1)[0].strip()
                else:
                    poem = text

                return poem

        # Fallback: unexpected format
        return f"Error: Unexpected API response format: {result}"

    except requests.exceptions.RequestException as e:
        return f"Error: Failed to connect to API: {str(e)}"
    except Exception as e:
        return f"Error: An unexpected error occurred: {str(e)}"


def test_api_connection(api_url: Optional[str] = None, api_token: Optional[str] = None) -> bool:
    """
    Test the connection to the API.
    
    Args:
        api_url (str, optional): Custom API URL
        api_token (str, optional): API token for authentication
        
    Returns:
        bool: True if connection is successful, False otherwise
    """
    if not api_token:
        api_token = os.getenv('HF_API_TOKEN')
    
    if not api_token:
        return False
    
    if not api_url:
        api_url = os.getenv('HF_API_URL', 'https://api-inference.huggingface.co/models/')
    
    headers = {
        "Authorization": f"Bearer {api_token}"
    }
    
    try:
        response = requests.get(api_url, headers=headers)
        return response.status_code == 200
    except:
        return False