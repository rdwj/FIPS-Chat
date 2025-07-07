"""Configuration management for the FIPS Chat application."""

import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class AppConfig:
    """Main application configuration."""
    # API settings
    default_api_endpoint: Optional[str] = None
    default_api_key: Optional[str] = None
    default_api_provider: str = "openai_compatible"
    request_timeout: int = 60
    
    # File handling
    max_file_size_mb: int = 10
    supported_image_formats: List[str] = field(default_factory=lambda: ["jpg", "jpeg", "png", "webp", "gif"])
    
    # Model generation parameters
    temperature: float = 0.7
    max_tokens: int = 2048
    
    # UI settings
    theme: str = "light"
    chat_history_limit: int = 50
    
    # Model discovery settings
    model_cache_ttl: int = 300  # 5 minutes
    auto_discover_models: bool = True


# API endpoint templates for common providers
API_TEMPLATES = {
    "openai": {
        "name": "OpenAI",
        "endpoint": "https://api.openai.com/v1",
        "requires_key": True,
        "example_models": ["gpt-4", "gpt-3.5-turbo", "gpt-4-vision-preview"]
    },
    "anthropic": {
        "name": "Anthropic Claude",
        "endpoint": "https://api.anthropic.com",
        "requires_key": True,
        "example_models": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"]
    },
    "vllm": {
        "name": "vLLM Server",
        "endpoint": "http://localhost:8000/v1",
        "requires_key": False,
        "example_models": ["Auto-discovered from endpoint"]
    },
    "ollama": {
        "name": "Ollama API",
        "endpoint": "http://localhost:11434/v1",
        "requires_key": False,
        "example_models": ["Auto-discovered from endpoint"]
    },
    "text-generation-webui": {
        "name": "Text Generation WebUI",
        "endpoint": "http://localhost:5000/v1",
        "requires_key": False,
        "example_models": ["Auto-discovered from endpoint"]
    }
}


def get_config() -> AppConfig:
    """Get application configuration, allowing environment variable overrides."""
    config = AppConfig()
    
    # Override with environment variables if present
    config.default_api_endpoint = os.getenv("API_ENDPOINT", config.default_api_endpoint)
    config.default_api_key = os.getenv("API_KEY", config.default_api_key)
    config.default_api_provider = os.getenv("API_PROVIDER", config.default_api_provider)
    
    if os.getenv("REQUEST_TIMEOUT"):
        config.request_timeout = int(os.getenv("REQUEST_TIMEOUT"))
    
    if os.getenv("MAX_FILE_SIZE_MB"):
        config.max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB"))
    
    if os.getenv("TEMPERATURE"):
        config.temperature = float(os.getenv("TEMPERATURE"))
    
    if os.getenv("MAX_TOKENS"):
        config.max_tokens = int(os.getenv("MAX_TOKENS"))
    
    if os.getenv("MODEL_CACHE_TTL"):
        config.model_cache_ttl = int(os.getenv("MODEL_CACHE_TTL"))
    
    if os.getenv("AUTO_DISCOVER_MODELS"):
        config.auto_discover_models = os.getenv("AUTO_DISCOVER_MODELS").lower() in ("true", "1", "yes")
    
    return config


def get_api_template(template_name: str) -> Optional[Dict[str, Any]]:
    """Get API template configuration."""
    return API_TEMPLATES.get(template_name)


def get_all_api_templates() -> Dict[str, Dict[str, Any]]:
    """Get all available API templates."""
    return API_TEMPLATES.copy()


def is_vision_capable_model(model_name: str) -> bool:
    """Check if a model likely supports vision/image analysis based on name."""
    name_lower = model_name.lower()
    
    # Common vision model indicators
    vision_indicators = [
        "vision", "llava", "claude-3", "gpt-4-vision", "gpt-4o", 
        "gpt-4-turbo", "gemini-pro-vision", "pixtral", "qwen-vl"
    ]
    
    return any(indicator in name_lower for indicator in vision_indicators)


def is_code_capable_model(model_name: str) -> bool:
    """Check if a model is optimized for code generation."""
    name_lower = model_name.lower()
    
    # Common code model indicators
    code_indicators = [
        "code", "coder", "codellama", "starcoder", "deepseek-coder",
        "wizard-coder", "magicoder", "granite-code"
    ]
    
    return any(indicator in name_lower for indicator in code_indicators)