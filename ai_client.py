"""Unified AI client for multiple providers with dynamic model discovery."""

import streamlit as st
import requests
import json
import base64
import time
import uuid
from typing import Dict, List, Optional, Generator, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class APIType(Enum):
    """Supported API types."""
    OPENAI_COMPATIBLE = "openai_compatible"
    ANTHROPIC = "anthropic"
    AGENTIC_PIPELINE = "agentic_pipeline"
    CUSTOM = "custom"


class ModelCapability(Enum):
    """Model capabilities."""
    CHAT = "chat"
    VISION = "vision"
    CODE = "code"
    EMBEDDING = "embedding"


@dataclass
class ModelInfo:
    """Information about a discovered model."""
    id: str
    name: str
    capabilities: List[ModelCapability] = field(default_factory=list)
    context_length: Optional[int] = None
    size: Optional[str] = None
    description: Optional[str] = None
    provider_specific: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIEndpoint:
    """Configuration for an API endpoint."""
    url: str
    api_key: Optional[str] = None
    api_type: APIType = APIType.OPENAI_COMPATIBLE
    timeout: int = 60
    headers: Dict[str, str] = field(default_factory=dict)


class UnifiedAPIClient:
    """Unified client for multiple AI API providers."""
    
    def __init__(self, endpoint: APIEndpoint):
        self.endpoint = endpoint
        self.session = requests.Session()
        self._models_cache: Dict[str, List[ModelInfo]] = {}
        self._cache_timestamp: Dict[str, float] = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Set up authentication headers
        self._setup_authentication()
    
    def _setup_authentication(self):
        """Set up authentication headers based on API type."""
        if not self.endpoint.api_key:
            return
            
        if self.endpoint.api_type == APIType.OPENAI_COMPATIBLE:
            self.session.headers.update({
                "Authorization": f"Bearer {self.endpoint.api_key}",
                "Content-Type": "application/json"
            })
        elif self.endpoint.api_type == APIType.ANTHROPIC:
            self.session.headers.update({
                "x-api-key": self.endpoint.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            })
        elif self.endpoint.api_type == APIType.AGENTIC_PIPELINE:
            if self.endpoint.api_key:
                self.session.headers.update({
                    "Authorization": f"Bearer {self.endpoint.api_key}",
                    "Content-Type": "application/json"
                })
        
        # Add any custom headers
        self.session.headers.update(self.endpoint.headers)
    
    def test_connection(self) -> Tuple[bool, str]:
        """Test connection to the API endpoint."""
        try:
            if self.endpoint.api_type == APIType.AGENTIC_PIPELINE:
                return self._test_agentic_pipeline()
            elif self.endpoint.api_type == APIType.OPENAI_COMPATIBLE:
                return self._test_openai_compatible()
            elif self.endpoint.api_type == APIType.ANTHROPIC:
                return self._test_anthropic()
            else:
                return self._test_custom_api()
                
        except requests.exceptions.RequestException as e:
            return False, f"Connection error: {str(e)}"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"
    
    def _test_openai_compatible(self) -> Tuple[bool, str]:
        """Test OpenAI-compatible API connection."""
        # Try models endpoint first
        try:
            models_url = f"{self.endpoint.url.rstrip('/')}/models"
            response = self.session.get(models_url, timeout=10)
            
            if response.status_code == 200:
                models_data = response.json()
                if "data" in models_data:
                    model_count = len(models_data["data"])
                    return True, f"Connected successfully. {model_count} models available."
                return True, "Connected successfully."
            elif response.status_code == 404:
                # Models endpoint not available, try chat completion with dummy model
                return self._test_chat_completion("test-model")
            else:
                return False, f"API returned status {response.status_code}"
                
        except Exception:
            # Fallback to chat completion test
            return self._test_chat_completion("gpt-3.5-turbo")
    
    def _test_chat_completion(self, model_name: str) -> Tuple[bool, str]:
        """Test with a simple chat completion."""
        chat_url = f"{self.endpoint.url.rstrip('/')}/chat/completions"
        
        test_payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 1,
            "temperature": 0
        }
        
        response = self.session.post(chat_url, json=test_payload, timeout=10)
        
        if response.status_code == 200:
            return True, f"Chat completion test successful"
        elif response.status_code == 401:
            return False, "Authentication failed. Check your API key."
        elif response.status_code == 404:
            return False, "Chat endpoint not found or model unavailable."
        else:
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", response.text[:200])
                return False, f"API error: {error_msg}"
            except:
                return False, f"API error: {response.status_code} - {response.text[:200]}"
    
    def _test_anthropic(self) -> Tuple[bool, str]:
        """Test Anthropic API connection."""
        url = f"{self.endpoint.url.rstrip('/')}/messages"
        
        test_payload = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "test"}]
        }
        
        response = self.session.post(url, json=test_payload, timeout=10)
        
        if response.status_code == 200:
            return True, "Anthropic API connection successful"
        elif response.status_code == 401:
            return False, "Authentication failed. Check your API key."
        else:
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", response.text[:200])
                return False, f"API error: {error_msg}"
            except:
                return False, f"API error: {response.status_code} - {response.text[:200]}"
    
    def _test_agentic_pipeline(self) -> Tuple[bool, str]:
        """Test agentic pipeline connection."""
        # Try health endpoints first
        health_endpoints = [
            f"{self.endpoint.url.rstrip('/')}/health",
            f"{self.endpoint.url.rstrip('/')}/status", 
            f"{self.endpoint.url.rstrip('/')}/ping"
        ]
        
        for health_url in health_endpoints:
            try:
                response = self.session.get(health_url, timeout=5)
                if response.status_code == 200:
                    return True, f"Agentic pipeline is healthy"
            except:
                continue
        
        # Fallback to main endpoint test
        try:
            test_payload = {
                "message": "test",
                "session_id": "test-connection"
            }
            
            response = self.session.post(self.endpoint.url, json=test_payload, timeout=10)
            
            if response.status_code == 200:
                return True, "Agentic pipeline endpoint is responding"
            elif response.status_code == 401:
                return False, "Authentication required. Check your API key."
            elif response.status_code == 404:
                return False, "Endpoint not found. Check your URL."
            else:
                return False, f"Pipeline returned status {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return False, "Could not connect to agentic pipeline. Check endpoint URL."
        except requests.exceptions.Timeout:
            return False, "Connection timeout. Pipeline may be slow to respond."
        except Exception as e:
            return False, f"Pipeline test failed: {str(e)}"
    
    def _test_custom_api(self) -> Tuple[bool, str]:
        """Test custom API with basic connectivity check."""
        try:
            response = self.session.get(self.endpoint.url, timeout=10)
            if response.status_code < 400:
                return True, "Custom API endpoint is reachable"
            else:
                return False, f"Custom API returned status {response.status_code}"
        except Exception as e:
            return False, f"Custom API test failed: {str(e)}"
    
    def discover_models(self, force_refresh: bool = False) -> List[ModelInfo]:
        """Discover available models from the API endpoint."""
        cache_key = self.endpoint.url
        
        # Check cache
        if not force_refresh and cache_key in self._models_cache:
            if time.time() - self._cache_timestamp.get(cache_key, 0) < self.cache_ttl:
                return self._models_cache[cache_key]
        
        try:
            if self.endpoint.api_type == APIType.OPENAI_COMPATIBLE:
                models = self._discover_openai_models()
            elif self.endpoint.api_type == APIType.ANTHROPIC:
                models = self._discover_anthropic_models()
            elif self.endpoint.api_type == APIType.AGENTIC_PIPELINE:
                models = self._discover_agentic_models()
            else:
                models = self._discover_custom_models()
            
            # Cache the results
            self._models_cache[cache_key] = models
            self._cache_timestamp[cache_key] = time.time()
            
            return models
            
        except Exception as e:
            logger.error(f"Model discovery failed: {str(e)}")
            return []
    
    def _discover_openai_models(self) -> List[ModelInfo]:
        """Discover models from OpenAI-compatible API."""
        models_url = f"{self.endpoint.url.rstrip('/')}/models"
        
        try:
            response = self.session.get(models_url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            models = []
            
            for model_data in data.get("data", []):
                model_id = model_data.get("id", "")
                if not model_id:
                    continue
                
                # Infer capabilities from model name
                capabilities = self._infer_capabilities_from_name(model_id)
                
                model_info = ModelInfo(
                    id=model_id,
                    name=model_data.get("name", model_id),
                    capabilities=capabilities,
                    context_length=model_data.get("context_length"),
                    provider_specific=model_data
                )
                models.append(model_info)
            
            return models
            
        except Exception as e:
            logger.warning(f"Could not discover OpenAI models: {str(e)}")
            return []
    
    def _discover_anthropic_models(self) -> List[ModelInfo]:
        """Discover Anthropic models (hardcoded list since no discovery endpoint)."""
        # Anthropic doesn't have a models discovery endpoint, so we use known models
        known_models = [
            ("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet", [ModelCapability.CHAT, ModelCapability.VISION]),
            ("claude-3-5-haiku-20241022", "Claude 3.5 Haiku", [ModelCapability.CHAT, ModelCapability.VISION]),
            ("claude-3-opus-20240229", "Claude 3 Opus", [ModelCapability.CHAT, ModelCapability.VISION]),
            ("claude-3-sonnet-20240229", "Claude 3 Sonnet", [ModelCapability.CHAT, ModelCapability.VISION]),
            ("claude-3-haiku-20240307", "Claude 3 Haiku", [ModelCapability.CHAT, ModelCapability.VISION]),
        ]
        
        models = []
        for model_id, name, capabilities in known_models:
            models.append(ModelInfo(
                id=model_id,
                name=name,
                capabilities=capabilities,
                context_length=200000 if "claude-3" in model_id else None
            ))
        
        return models
    
    def _discover_agentic_models(self) -> List[ModelInfo]:
        """Discover models from agentic pipeline (may not have models concept)."""
        # Agentic pipelines typically don't expose models, they're more like services
        # We create a default "pipeline" model
        return [ModelInfo(
            id="default-pipeline",
            name="Agentic Pipeline",
            capabilities=[ModelCapability.CHAT],
            description="Agentic pipeline endpoint"
        )]
    
    def _discover_custom_models(self) -> List[ModelInfo]:
        """Attempt to discover models from custom API."""
        # Try common model endpoints
        possible_endpoints = ["/models", "/v1/models", "/api/models"]
        
        for endpoint in possible_endpoints:
            try:
                models_url = f"{self.endpoint.url.rstrip('/')}{endpoint}"
                response = self.session.get(models_url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if "data" in data:
                        return self._parse_openai_format_models(data)
                    elif isinstance(data, list):
                        return self._parse_simple_model_list(data)
                        
            except:
                continue
        
        # If no models endpoint found, return empty list
        return []
    
    def _parse_openai_format_models(self, data: Dict) -> List[ModelInfo]:
        """Parse OpenAI-format model data."""
        models = []
        for model_data in data.get("data", []):
            if isinstance(model_data, dict) and "id" in model_data:
                capabilities = self._infer_capabilities_from_name(model_data["id"])
                models.append(ModelInfo(
                    id=model_data["id"],
                    name=model_data.get("name", model_data["id"]),
                    capabilities=capabilities,
                    provider_specific=model_data
                ))
        return models
    
    def _parse_simple_model_list(self, data: List) -> List[ModelInfo]:
        """Parse simple list of model names."""
        models = []
        for item in data:
            if isinstance(item, str):
                capabilities = self._infer_capabilities_from_name(item)
                models.append(ModelInfo(
                    id=item,
                    name=item,
                    capabilities=capabilities
                ))
            elif isinstance(item, dict) and "name" in item:
                name = item["name"]
                capabilities = self._infer_capabilities_from_name(name)
                models.append(ModelInfo(
                    id=name,
                    name=name,
                    capabilities=capabilities,
                    provider_specific=item
                ))
        return models
    
    def _infer_capabilities_from_name(self, model_name: str) -> List[ModelCapability]:
        """Infer model capabilities from model name."""
        name_lower = model_name.lower()
        capabilities = []
        
        # Embedding models first (most specific)
        if any(keyword in name_lower for keyword in ["embed", "embedding", "text-embedding"]):
            capabilities = [ModelCapability.EMBEDDING]
            return capabilities
        
        # All other models get chat capability by default
        capabilities.append(ModelCapability.CHAT)
        
        # Vision models
        if any(keyword in name_lower for keyword in [
            "vision", "llava", "claude-3", "gpt-4-vision", "gpt-4o", "gpt-4-turbo",
            "minicpm-v", "qwen-vl", "internvl", "cogvlm", "blip", "instructblip"
        ]):
            capabilities.append(ModelCapability.VISION)
        
        # Code models
        if any(keyword in name_lower for keyword in [
            "code", "coder", "codellama", "starcoder", "deepseek-coder", 
            "phind-codellama", "wizardcoder", "magicoder", "granite-code"
        ]):
            capabilities.append(ModelCapability.CODE)
        
        return capabilities
    
    def chat(self, model: str, messages: List[Dict[str, str]], stream: bool = True, **kwargs) -> Generator[str, None, None]:
        """Send a chat request and yield response chunks."""
        try:
            if self.endpoint.api_type == APIType.OPENAI_COMPATIBLE:
                yield from self._chat_openai_compatible(model, messages, stream, **kwargs)
            elif self.endpoint.api_type == APIType.ANTHROPIC:
                yield from self._chat_anthropic(model, messages, **kwargs)
            elif self.endpoint.api_type == APIType.AGENTIC_PIPELINE:
                yield from self._chat_agentic_pipeline(messages, **kwargs)
            else:
                yield "Error: Unsupported API type for chat"
                
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def _chat_openai_compatible(self, model: str, messages: List[Dict[str, str]], stream: bool = True, **kwargs) -> Generator[str, None, None]:
        """Chat using OpenAI-compatible API."""
        url = f"{self.endpoint.url.rstrip('/')}/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2048)
        }
        
        response = self.session.post(url, json=payload, stream=stream, timeout=self.endpoint.timeout)
        response.raise_for_status()
        
        if stream:
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        line_str = line_str[6:]
                    
                    if line_str.strip() == '[DONE]':
                        break
                    
                    try:
                        chunk = json.loads(line_str)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                    except json.JSONDecodeError:
                        continue
        else:
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0]["message"]["content"]
                yield content
    
    def _chat_anthropic(self, model: str, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        """Chat using Anthropic API."""
        url = f"{self.endpoint.url.rstrip('/')}/messages"
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 0.7)
        }
        
        response = self.session.post(url, json=payload, timeout=self.endpoint.timeout)
        response.raise_for_status()
        
        data = response.json()
        if "content" in data and len(data["content"]) > 0:
            yield data["content"][0]["text"]
    
    def _chat_agentic_pipeline(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        """Chat using agentic pipeline."""
        # Get the latest user message
        user_message = messages[-1]["content"] if messages else ""
        
        # Get or generate session ID
        session_id = st.session_state.get("agent_session_id", str(uuid.uuid4()))
        st.session_state.agent_session_id = session_id
        
        payload = {
            "message": user_message,
            "session_id": session_id
        }
        
        # Add conversation history if supported
        if len(messages) > 1:
            payload["conversation_history"] = messages[:-1]
        
        response = self.session.post(self.endpoint.url, json=payload, timeout=self.endpoint.timeout)
        response.raise_for_status()
        
        data = response.json()
        
        # Handle different response formats
        if "response" in data:
            yield data["response"]
        elif "message" in data:
            yield data["message"]
        elif "content" in data:
            yield data["content"]
        elif "text" in data:
            yield data["text"]
        else:
            # Try to find any string value
            for value in data.values():
                if isinstance(value, str) and len(value) > 0:
                    yield value
                    return
            yield "Error: Could not parse response from agentic pipeline"
    
    def generate_with_image(self, model: str, prompt: str, image_data: bytes, stream: bool = True, **kwargs) -> Generator[str, None, None]:
        """Generate text response with image input."""
        try:
            if self.endpoint.api_type == APIType.OPENAI_COMPATIBLE:
                yield from self._generate_image_openai_compatible(model, prompt, image_data, stream, **kwargs)
            elif self.endpoint.api_type == APIType.ANTHROPIC:
                yield from self._generate_image_anthropic(model, prompt, image_data, **kwargs)
            else:
                yield "Error: Image analysis not supported for this API type"
                
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def _generate_image_openai_compatible(self, model: str, prompt: str, image_data: bytes, stream: bool = True, **kwargs) -> Generator[str, None, None]:
        """Generate image response using OpenAI-compatible API."""
        # Encode image to base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Try different formats based on the API
        formats_to_try = [
            # Format 1: OpenAI GPT-4 Vision style
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]
                    }
                ],
                "stream": stream,
                "max_tokens": kwargs.get("max_tokens", 2048),
                "temperature": kwargs.get("temperature", 0.7)
            },
            # Format 2: Some APIs use "images" field
            {
                "model": model,
                "prompt": prompt,
                "images": [image_base64],
                "stream": stream,
                "max_tokens": kwargs.get("max_tokens", 2048),
                "temperature": kwargs.get("temperature", 0.7)
            }
        ]
        
        last_error = None
        for payload in formats_to_try:
            try:
                # Try chat/completions first, then generate
                endpoints_to_try = ["/chat/completions", "/generate"]
                
                for endpoint in endpoints_to_try:
                    try:
                        url = f"{self.endpoint.url.rstrip('/')}{endpoint}"
                        response = self.session.post(url, json=payload, stream=stream, timeout=self.endpoint.timeout)
                        
                        if response.status_code == 200:
                            if stream:
                                for line in response.iter_lines():
                                    if line:
                                        line_str = line.decode('utf-8')
                                        if line_str.startswith('data: '):
                                            line_str = line_str[6:]
                                        
                                        if line_str.strip() == '[DONE]':
                                            break
                                        
                                        try:
                                            chunk = json.loads(line_str)
                                            # Handle both chat and generate responses
                                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                                delta = chunk["choices"][0].get("delta", {})
                                                if "content" in delta:
                                                    yield delta["content"]
                                            elif "response" in chunk:
                                                yield chunk["response"]
                                        except json.JSONDecodeError:
                                            continue
                            else:
                                data = response.json()
                                if "choices" in data and len(data["choices"]) > 0:
                                    content = data["choices"][0]["message"]["content"]
                                    yield content
                                elif "response" in data:
                                    yield data["response"]
                            return  # Success, exit all loops
                    except Exception as e:
                        last_error = e
                        continue
            except Exception as e:
                last_error = e
                continue
        
        # If all formats failed
        if last_error:
            yield f"Error: Could not process image with this API: {str(last_error)}"
        else:
            yield "Error: Image analysis not supported by this endpoint"
    
    def _generate_image_anthropic(self, model: str, prompt: str, image_data: bytes, **kwargs) -> Generator[str, None, None]:
        """Generate image response using Anthropic API."""
        # Encode image to base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        url = f"{self.endpoint.url.rstrip('/')}/messages"
        
        payload = {
            "model": model,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        }
        
        response = self.session.post(url, json=payload, timeout=self.endpoint.timeout)
        response.raise_for_status()
        
        data = response.json()
        if "content" in data and len(data["content"]) > 0:
            yield data["content"][0]["text"]
    
    def get_models_by_capability(self, capability: ModelCapability) -> List[ModelInfo]:
        """Get models that support a specific capability."""
        all_models = self.discover_models()
        return [model for model in all_models if capability in model.capabilities]
    
    def chat_with_rag(
        self,
        query: str,
        model: str,
        rag_pipeline,  # RAGPipeline from rag.rag_pipeline
        max_sources: int = 5,
        relevance_threshold: float = 0.1,
        stream: bool = True,
        **kwargs
    ):
        """
        Chat with RAG enhancement using the RAG pipeline.
        
        Args:
            query: User query
            model: AI model to use
            rag_pipeline: RAGPipeline instance for context retrieval
            max_sources: Maximum number of sources to use
            relevance_threshold: Minimum relevance score for sources
            stream: Whether to stream the response
            **kwargs: Additional arguments for chat generation
            
        Returns:
            RAGResponse object with enhanced response and sources
        """
        return rag_pipeline.process_rag_query(
            query=query,
            model=model,
            max_sources=max_sources,
            relevance_threshold=relevance_threshold,
            stream=stream,
            **kwargs
        )
    
    def _format_rag_messages(
        self,
        query: str,
        context: str,
        sources: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Format messages for RAG-enhanced chat."""
        # This is used internally by RAGPipeline, keeping it here for potential future use
        sources_text = "\n".join([
            f"Source {i+1}: {source['document']}, Page {source['page_number']}"
            for i, source in enumerate(sources)
        ])
        
        rag_prompt = f"""Based on the following context, please answer the user's question.

Context:
{context}

Sources:
{sources_text}

Question: {query}

Please provide a comprehensive answer and cite your sources using [Source N] format."""
        
        return [{"role": "user", "content": rag_prompt}]


def get_ai_client() -> Optional[UnifiedAPIClient]:
    """Get configured AI client from session state."""
    endpoint_url = st.session_state.get("api_endpoint")
    if not endpoint_url:
        return None
    
    api_key = st.session_state.get("api_key")
    provider = st.session_state.get("api_provider", "openai_compatible")
    
    # Map provider to API type
    if provider == "agentic_pipeline":
        api_type = APIType.AGENTIC_PIPELINE
    elif provider == "anthropic":
        api_type = APIType.ANTHROPIC
    elif provider == "openai_compatible":
        api_type = APIType.OPENAI_COMPATIBLE
    else:
        api_type = APIType.CUSTOM
    
    endpoint = APIEndpoint(
        url=endpoint_url,
        api_key=api_key,
        api_type=api_type
    )
    
    return UnifiedAPIClient(endpoint)


def test_api_connection(endpoint_url: str, api_key: Optional[str] = None, provider: str = "openai_compatible") -> Tuple[bool, str]:
    """Test API connection with given parameters."""
    # Map provider to API type
    if provider == "agentic_pipeline":
        api_type = APIType.AGENTIC_PIPELINE
    elif provider == "anthropic":
        api_type = APIType.ANTHROPIC
    elif provider == "openai_compatible":
        api_type = APIType.OPENAI_COMPATIBLE
    else:
        api_type = APIType.CUSTOM
    
    endpoint = APIEndpoint(
        url=endpoint_url,
        api_key=api_key,
        api_type=api_type
    )
    
    client = UnifiedAPIClient(endpoint)
    return client.test_connection()