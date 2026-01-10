import os
import json
from typing import Optional
import requests
from datetime import datetime


class GroqClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "mixtral-8x7b-32768"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY must be set in environment or passed to constructor")
        
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def send_extraction_request(self, prompt: str, temperature: float = 0.1) -> dict:
        """Send extraction request to Groq API."""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise knowledge extraction engine. Output only valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": 4096,
            "top_p": 1,
            "stream": False
        }
        
        print("\n" + "="*80)
        print("GROQ API REQUEST")
        print("="*80)
        print(f"Model: {self.model}")
        print(f"Temperature: {temperature}")
        print(f"\nPrompt (first 500 chars):\n{prompt[:500]}...")
        print("="*80 + "\n")
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            
            print("\n" + "="*80)
            print("GROQ API RESPONSE")
            print("="*80)
            print(f"Status Code: {response.status_code}")
            print(f"Model Used: {result.get('model', 'N/A')}")
            print(f"Total Tokens: {result.get('usage', {}).get('total_tokens', 'N/A')}")
            print(f"\nResponse Content:\n{result['choices'][0]['message']['content']}")
            print("="*80 + "\n")
            
            return {
                "success": True,
                "content": result["choices"][0]["message"]["content"],
                "usage": result.get("usage", {}),
                "model": result.get("model"),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Groq API request failed: {str(e)}"
            print(f"\n❌ ERROR: {error_msg}\n")
            
            return {
                "success": False,
                "error": error_msg,
                "timestamp": datetime.utcnow().isoformat()
            }
