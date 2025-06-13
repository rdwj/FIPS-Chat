"""Ollama API client for the Streamlit application."""

import requests
import json
import base64
from typing import Dict, List, Optional, Generator, Any
from dataclasses import dataclass
import time
import streamlit as st

from config import AppConfig, get_config


@dataclass
class ModelInfo:
    """Information about an Ollama model."""
    name: str
    size: int
    digest: str
    modified_at: str
    details: Dict[str, Any]


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_config()
        self.base_url = self.config.ollama_host
        self.session = requests.Session()
        self.session.timeout = self.config.request_timeout
    
    def _make_request(self, endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Dict:
        """Make a request to the Ollama API with error handling."""
        url = f"{self.base_url}/api/{endpoint}"
        
        try:
            if method == "GET":
                response = self.session.get(url)
            elif method == "POST":
                response = self.session.post(url, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Could not connect to Ollama at {self.base_url}. Make sure Ollama is running.")
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request to Ollama timed out after {self.config.request_timeout} seconds.")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise ValueError(f"Model or endpoint not found: {endpoint}")
            else:
                raise RuntimeError(f"Ollama API error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error connecting to Ollama: {str(e)}")
    
    def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models from Ollama."""
        try:
            response = self._make_request("tags")
            models = []
            
            for model_data in response.get("models", []):
                models.append(ModelInfo(
                    name=model_data["name"],
                    size=model_data.get("size", 0),
                    digest=model_data.get("digest", ""),
                    modified_at=model_data.get("modified_at", ""),
                    details=model_data.get("details", {})
                ))
            
            return models
        
        except Exception as e:
            st.error(f"Failed to get available models: {str(e)}")
            return []
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get detailed information about a specific model."""
        try:
            response = self._make_request(f"show", "POST", {"name": model_name})
            return response
        except Exception as e:
            st.error(f"Failed to get model info for {model_name}: {str(e)}")
            return None
    
    def chat(self, model: str, messages: List[Dict[str, str]], stream: bool = True) -> Generator[str, None, None]:
        """Send a chat request to Ollama and yield response chunks."""
        data = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        }
        
        try:
            url = f"{self.base_url}/api/chat"
            response = self.session.post(url, json=data, stream=stream, timeout=self.config.request_timeout)
            response.raise_for_status()
            
            if stream:
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            if chunk.get("message", {}).get("content"):
                                content = chunk["message"]["content"]
                                full_response += content
                                yield content
                            
                            if chunk.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                result = response.json()
                if result.get("message", {}).get("content"):
                    yield result["message"]["content"]
        
        except requests.exceptions.ConnectionError:
            yield "Error: Could not connect to Ollama. Please make sure Ollama is running."
        except requests.exceptions.Timeout:
            yield "Error: Request timed out. The model might be taking too long to respond."
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def generate_with_image(self, model: str, prompt: str, image_data: bytes, stream: bool = True) -> Generator[str, None, None]:
        """Generate text response with image input."""
        # Encode image to base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        data = {
            "model": model,
            "prompt": prompt,
            "images": [image_base64],
            "stream": stream,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        }
        
        try:
            url = f"{self.base_url}/api/generate"
            response = self.session.post(url, json=data, stream=stream, timeout=self.config.request_timeout)
            response.raise_for_status()
            
            if stream:
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            if chunk.get("response"):
                                yield chunk["response"]
                            
                            if chunk.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                result = response.json()
                if result.get("response"):
                    yield result["response"]
        
        except requests.exceptions.ConnectionError:
            yield "Error: Could not connect to Ollama. Please make sure Ollama is running."
        except requests.exceptions.Timeout:
            yield "Error: Request timed out. The model might be taking too long to respond."
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available."""
        available_models = self.get_available_models()
        return any(model.name == model_name for model in available_models)
    
    def pull_model(self, model_name: str) -> Generator[str, None, None]:
        """Pull a model from Ollama registry."""
        data = {"name": model_name}
        
        try:
            url = f"{self.base_url}/api/pull"
            response = self.session.post(url, json=data, stream=True, timeout=600)  # Longer timeout for pulling
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        if chunk.get("status"):
                            yield chunk["status"]
                        
                        if chunk.get("completed"):
                            break
                    except json.JSONDecodeError:
                        continue
        
        except Exception as e:
            yield f"Error pulling model: {str(e)}"


def get_ollama_client() -> OllamaClient:
    """Get a configured Ollama client instance."""
    if "ollama_client" not in st.session_state:
        st.session_state.ollama_client = OllamaClient()
    return st.session_state.ollama_client