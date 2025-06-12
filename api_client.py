"""API client for external AI services (OpenAI-compatible, custom APIs)."""

import streamlit as st
import requests
import json
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class APIConfig:
    """Configuration for API connection."""
    provider: str
    endpoint: str
    api_key: Optional[str] = None
    api_type: str = "openai_compatible"
    model_name: Optional[str] = None


class ExternalAPIClient:
    """Client for external AI APIs."""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.session = requests.Session()
        
        # Set headers based on API type
        if config.api_key:
            if config.api_type == "openai_compatible":
                self.session.headers.update({
                    "Authorization": f"Bearer {config.api_key}",
                    "Content-Type": "application/json"
                })
            elif config.api_type == "anthropic":
                self.session.headers.update({
                    "x-api-key": config.api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                })
    
    def test_connection(self, model_name: Optional[str] = None) -> Tuple[bool, str]:
        """Test API connection."""
        try:
            if self.config.provider == "agentic_pipeline":
                return self._test_agentic_pipeline()
            elif self.config.api_type == "openai_compatible":
                return self._test_openai_compatible(model_name)
            elif self.config.api_type == "anthropic":
                return self._test_anthropic(model_name)
            else:
                return self._test_custom_api(model_name)
                
        except requests.exceptions.RequestException as e:
            return False, f"Connection error: {str(e)}"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"
    
    def _test_openai_compatible(self, model_name: Optional[str] = None) -> Tuple[bool, str]:
        """Test OpenAI-compatible API."""
        # Try to list models first
        try:
            models_url = f"{self.config.endpoint.rstrip('/')}/models"
            response = self.session.get(models_url, timeout=10)
            
            if response.status_code == 200:
                models_data = response.json()
                if "data" in models_data:
                    available_models = [model["id"] for model in models_data["data"]]
                    if model_name and model_name not in available_models:
                        return False, f"Model '{model_name}' not found. Available: {', '.join(available_models[:5])}"
                    return True, f"Connected successfully. {len(available_models)} models available."
                return True, "Connected successfully."
            else:
                # If models endpoint fails, try a simple chat completion
                return self._test_chat_completion(model_name or "gpt-3.5-turbo")
                
        except Exception:
            # Fallback to chat completion test
            return self._test_chat_completion(model_name or "gpt-3.5-turbo")
    
    def _test_chat_completion(self, model_name: str) -> Tuple[bool, str]:
        """Test with a simple chat completion."""
        chat_url = f"{self.config.endpoint.rstrip('/')}/chat/completions"
        
        test_payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 1,
            "temperature": 0
        }
        
        response = self.session.post(chat_url, json=test_payload, timeout=10)
        
        if response.status_code == 200:
            return True, f"Chat completion test successful with model '{model_name}'"
        elif response.status_code == 401:
            return False, "Authentication failed. Check your API key."
        elif response.status_code == 404:
            return False, f"Model '{model_name}' not found or endpoint incorrect."
        else:
            return False, f"API error: {response.status_code} - {response.text[:200]}"
    
    def _test_anthropic(self, model_name: Optional[str] = None) -> Tuple[bool, str]:
        """Test Anthropic API."""
        # Anthropic uses a different endpoint structure
        url = f"{self.config.endpoint.rstrip('/')}/messages"
        
        test_payload = {
            "model": model_name or "claude-3-sonnet-20240229",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "test"}]
        }
        
        response = self.session.post(url, json=test_payload, timeout=10)
        
        if response.status_code == 200:
            return True, f"Anthropic API test successful with model '{model_name or 'default'}'"
        elif response.status_code == 401:
            return False, "Authentication failed. Check your API key."
        else:
            return False, f"API error: {response.status_code} - {response.text[:200]}"
    
    def _test_agentic_pipeline(self) -> Tuple[bool, str]:
        """Test agentic pipeline connection."""
        # First try a health/status endpoint
        try:
            # Try common health check endpoints
            health_endpoints = [
                f"{self.config.endpoint.rstrip('/')}/health",
                f"{self.config.endpoint.rstrip('/')}/status", 
                f"{self.config.endpoint.rstrip('/')}/ping"
            ]
            
            for health_url in health_endpoints:
                try:
                    response = self.session.get(health_url, timeout=5)
                    if response.status_code == 200:
                        return True, f"Agentic pipeline is healthy (via {health_url.split('/')[-1]})"
                except:
                    continue
            
            # If no health endpoint, try the main chat endpoint with a test message
            test_payload = {
                "message": "test",
                "session_id": "test-connection"
            }
            
            response = self.session.post(self.config.endpoint, json=test_payload, timeout=10)
            
            if response.status_code == 200:
                return True, "Agentic pipeline chat endpoint is responding"
            elif response.status_code == 401:
                return False, "Authentication required. Check your API key."
            elif response.status_code == 404:
                return False, "Chat endpoint not found. Check your endpoint URL."
            else:
                return False, f"Pipeline returned status {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return False, "Could not connect to agentic pipeline. Check endpoint URL."
        except requests.exceptions.Timeout:
            return False, "Connection timeout. Pipeline may be slow to respond."
        except Exception as e:
            return False, f"Pipeline test failed: {str(e)}"
    
    def _test_custom_api(self, model_name: Optional[str] = None) -> Tuple[bool, str]:
        """Test custom API with basic health check."""
        # Try a simple GET request first
        try:
            response = self.session.get(self.config.endpoint, timeout=10)
            if response.status_code < 400:
                return True, "Custom API endpoint is reachable"
            else:
                return False, f"Custom API returned status {response.status_code}"
        except Exception as e:
            return False, f"Custom API test failed: {str(e)}"
    
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response from the API."""
        if self.config.provider == "agentic_pipeline":
            return self._generate_agentic_pipeline(messages, **kwargs)
        elif self.config.api_type == "openai_compatible":
            return self._generate_openai_compatible(messages, **kwargs)
        elif self.config.api_type == "anthropic":
            return self._generate_anthropic(messages, **kwargs)
        else:
            raise NotImplementedError(f"Generation not implemented for {self.config.api_type}")
    
    def _generate_openai_compatible(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using OpenAI-compatible API."""
        url = f"{self.config.endpoint.rstrip('/')}/chat/completions"
        
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2048),
            "stream": False
        }
        
        response = self.session.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    def _generate_anthropic(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using Anthropic API."""
        url = f"{self.config.endpoint.rstrip('/')}/messages"
        
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 0.7)
        }
        
        response = self.session.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        return data["content"][0]["text"]
    
    def _generate_agentic_pipeline(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using agentic pipeline."""
        import streamlit as st
        import uuid
        
        # Get the latest user message
        user_message = messages[-1]["content"] if messages else ""
        
        # Get session ID from session state or generate one
        session_id = st.session_state.get("agent_session_id", "")
        if not session_id:
            session_id = str(uuid.uuid4())
            st.session_state.agent_session_id = session_id
        
        # Prepare payload for agentic pipeline
        payload = {
            "message": user_message,
            "session_id": session_id
        }
        
        # Add pipeline ID if specified
        pipeline_id = self.config.model_name
        if pipeline_id:
            payload["pipeline_id"] = pipeline_id
        
        # Add conversation history if the endpoint supports it
        if len(messages) > 1:
            payload["conversation_history"] = messages[:-1]  # All except the latest message
        
        # Add optional parameters
        if kwargs.get("temperature"):
            payload["temperature"] = kwargs["temperature"]
        if kwargs.get("max_tokens"):
            payload["max_tokens"] = kwargs["max_tokens"]
        
        response = self.session.post(self.config.endpoint, json=payload, timeout=120)
        response.raise_for_status()
        
        data = response.json()
        
        # Handle different response formats
        if "response" in data:
            return data["response"]
        elif "message" in data:
            return data["message"]
        elif "content" in data:
            return data["content"]
        elif "text" in data:
            return data["text"]
        else:
            # Fallback: try to find any string value in the response
            for key, value in data.items():
                if isinstance(value, str) and len(value) > 0:
                    return value
            
            raise ValueError(f"Could not parse response from agentic pipeline: {data}")


def get_api_client() -> Optional[ExternalAPIClient]:
    """Get configured API client from session state."""
    provider = st.session_state.get("api_provider")
    
    if provider == "ollama":
        return None
    
    endpoint = st.session_state.get("api_endpoint")
    api_key = st.session_state.get("api_key")
    model_name = st.session_state.get("external_model_name")
    api_type = st.session_state.get("api_type", "openai_compatible")
    
    if not endpoint:
        return None
    
    if provider == "agentic_pipeline":
        api_type = "agentic_pipeline"
    elif provider == "custom":
        api_type = api_type
    else:
        api_type = "openai_compatible"
    
    config = APIConfig(
        provider=provider,
        endpoint=endpoint,
        api_key=api_key,
        api_type=api_type,
        model_name=model_name
    )
    
    return ExternalAPIClient(config)


def test_api_connection(provider: str, endpoint: str, api_key: Optional[str] = None, 
                       model_name: Optional[str] = None) -> Tuple[bool, str]:
    """Test API connection with given parameters."""
    if provider == "agentic_pipeline":
        api_type = "agentic_pipeline"
    elif provider == "openai_compatible":
        api_type = "openai_compatible"
    else:
        api_type = provider
    
    config = APIConfig(
        provider=provider,
        endpoint=endpoint,
        api_key=api_key,
        api_type=api_type,
        model_name=model_name
    )
    
    client = ExternalAPIClient(config)
    return client.test_connection(model_name)