"""
API integration for remote poem generation
"""

import requests
import os
from typing import Optional


def generate_poem_via_api(style: str, seed: str, length: int = 50, api_url: Optional[str] = None, api_token: Optional[str] = None) -> str:
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
    # Use environment variable for API URL if not provided
    if not api_url:
        api_url = os.getenv('HF_API_URL', 'https://api-inference.huggingface.co/models/')
    
    # Use environment variable for API token if not provided
    if not api_token:
        api_token = os.getenv('HF_API_TOKEN')
    
    if not api_token:
        return "Error: No API token provided. Please set HF_API_TOKEN environment variable."
    
    # Construct the payload for the API request
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    # Prepare the prompt with style information
    style_prompts = {
        'haiku': f"Generate a haiku about {seed or 'nature'}:",
        'sonnet': f"Generate a sonnet about {seed or 'love'}:",
        'freeverse': f"Generate a free verse poem about {seed or 'life'}:"
    }
    
    prompt = style_prompts.get(style.lower(), f"Generate a {style} poem about {seed}:")
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": length,
            "temperature": 0.8,
            "top_p": 0.9,
            "repetition_penalty": 1.2
        }
    }
    
    try:
        # Make the API request
        response = requests.post(api_url, headers=headers, json=payload)
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            
            # Extract the generated text
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '')
                
                # Remove the original prompt from the response
                if prompt in generated_text:
                    poem = generated_text.replace(prompt, '').strip()
                else:
                    poem = generated_text.strip()
                
                return poem
            else:
                return f"Error: Unexpected API response format: {result}"
        else:
            return f"Error: API request failed with status {response.status_code}: {response.text}"
    
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