import time
import requests
from typing import List
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import Field

class GenericAPIEmbedding(BaseEmbedding):
    """
    A service-agnostic embedding class for RESTful APIs.
    Dynamically handles payload and response formats for HF and Google AI Studio.
    """
    api_url: str = Field(description="The full API endpoint URL")
    api_key: str = Field(description="The authentication key")
    model_name: str = Field(default="unknown", description="The model name for Ollama")
    max_retries: int = Field(default=5, description="Maximum number of retries for API calls")
    retry_delay: int = Field(default=5, description="Wait time between retries")
    
    def __init__(self, api_url: str, api_key: str, model_name: str = "unknown", max_retries: int = 5, retry_delay: int = 5, **kwargs):
        super().__init__(api_url=api_url, api_key=api_key, model_name=model_name, max_retries=max_retries, retry_delay=retry_delay, **kwargs)

    def _get_embedding(self, text: str) -> List[float]:
        # Detect service type from URL to adjust payload/headers
        is_google = "generativelanguage.googleapis.com" in self.api_url.lower()
        is_ollama = "localhost" in self.api_url or "127.0.0.1" in self.api_url
        
        max_retries = self.max_retries
        retry_delay = self.retry_delay

        if is_google:
            # Google AI Studio format
            headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
            payload = {"content": {"parts": [{"text": text}]}}
            
            for i in range(max_retries):
                try:
                    response = requests.post(self.api_url, headers=headers, json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        if isinstance(result, dict) and "embedding" in result:
                            return result["embedding"].get("values", [])
                        raise Exception(f"Unexpected Google response format: {result}")
                    elif response.status_code in [429, 503]:
                        print(f"  [API] Google Service Busy ({response.status_code}). Retrying in {retry_delay}s... ({i+1}/{max_retries})")
                        time.sleep(retry_delay)
                    else:
                        raise Exception(f"Google API Error: {response.status_code} - {response.text}")
                except Exception as e:
                    time.sleep(retry_delay)
        elif is_ollama:
            # Ollama API format (no key needed)
            headers = {"Content-Type": "application/json"}
            payload = {"model": self.model_name, "prompt": text}
            
            try:
                response = requests.post(self.api_url, headers=headers, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, dict) and "embedding" in result:
                        return result["embedding"]
                    raise Exception(f"Unexpected Ollama response format: {result}")
                else:
                    raise Exception(f"Ollama API Error: {response.status_code} - {response.text}")
            except Exception as e:
                raise e
        else:
            # Hugging Face Inference API / Generic
            # Mandatory header x-wait-for-model: true ensures the API waits for the model to load instead of 404/503
            headers = {
                "Authorization": f"Bearer {self.api_key}", 
                "Content-Type": "application/json",
                "x-wait-for-model": "true"
            }
            payload = {"inputs": text}
            
            for i in range(max_retries):
                try:
                    response = requests.post(self.api_url, headers=headers, json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        # Handle HF formats: flat list, nested list, or dict
                        if isinstance(result, list) and len(result) > 0:
                            return result[0] if isinstance(result[0], list) else result
                        if isinstance(result, dict) and "embedding" in result:
                            return result["embedding"]
                        return result
                    elif response.status_code in [503, 429, 404]:
                        # 404/503 can both mean the model is still being provisioned on the shared fleet
                        print(f"  [API] HF Service initializing/busy ({response.status_code}). Retrying in {retry_delay}s... ({i+1}/{max_retries})")
                        time.sleep(retry_delay)
                    else:
                        raise Exception(f"HF API Error: {response.status_code} - {response.text}")
                except Exception as e:
                    if i == max_retries - 1: raise e
                    time.sleep(retry_delay)
        
        return []

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_embedding(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)
