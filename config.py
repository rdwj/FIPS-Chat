"""Configuration management for the FIPS Chat application."""

import os
from typing import List
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for Ollama models."""
    name: str
    size: str
    type: str  # 'chat', 'vision', 'code'
    description: str


@dataclass
class AppConfig:
    """Main application configuration."""
    ollama_host: str = "http://localhost:11434"
    default_chat_model: str = "granite3.3:8b"
    default_vision_model: str = "llava:7b"
    request_timeout: int = 60
    max_file_size_mb: int = 10
    supported_image_formats: List[str] = field(default_factory=lambda: ["jpg", "jpeg", "png", "webp", "gif"])
    
    # Model generation parameters
    temperature: float = 0.7
    max_tokens: int = 2048
    
    # UI settings
    theme: str = "light"
    chat_history_limit: int = 50


# Recommended models based on specification
RECOMMENDED_MODELS = [
    ModelConfig("llava:7b", "4.7 GB", "vision", "Primary vision model for image description"),
    ModelConfig("granite3.2-vision:latest", "2.4 GB", "vision", "Alternative vision model"),
    ModelConfig("granite3.3:8b", "4.9 GB", "chat", "Primary chat model"),
    ModelConfig("gemma3:latest", "3.3 GB", "chat", "Lightweight chat alternative"),
    ModelConfig("phi4-mini:3.8b", "2.5 GB", "chat", "Fast response chat model"),
    ModelConfig("qwen2.5-coder:7b", "4.7 GB", "code", "Code-focused conversations"),
    ModelConfig("mistral-small3.1:24b", "15 GB", "chat", "High-quality responses (power users)"),
]


def get_config() -> AppConfig:
    """Get application configuration, allowing environment variable overrides."""
    config = AppConfig()
    
    # Override with environment variables if present
    config.ollama_host = os.getenv("OLLAMA_HOST", config.ollama_host)
    config.default_chat_model = os.getenv("DEFAULT_CHAT_MODEL", config.default_chat_model)
    config.default_vision_model = os.getenv("DEFAULT_VISION_MODEL", config.default_vision_model)
    
    if os.getenv("REQUEST_TIMEOUT"):
        config.request_timeout = int(os.getenv("REQUEST_TIMEOUT"))
    
    if os.getenv("MAX_FILE_SIZE_MB"):
        config.max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB"))
    
    if os.getenv("TEMPERATURE"):
        config.temperature = float(os.getenv("TEMPERATURE"))
    
    if os.getenv("MAX_TOKENS"):
        config.max_tokens = int(os.getenv("MAX_TOKENS"))
    
    return config


def get_model_info(model_name: str) -> ModelConfig:
    """Get information about a specific model."""
    for model in RECOMMENDED_MODELS:
        if model.name == model_name:
            return model
    
    # Return default info for unknown models
    return ModelConfig(model_name, "Unknown", "chat", "Model information not available")


def is_vision_model(model_name: str) -> bool:
    """Check if a model supports vision/image analysis."""
    model_info = get_model_info(model_name)
    return model_info.type == "vision" or "vision" in model_name.lower() or "llava" in model_name.lower()