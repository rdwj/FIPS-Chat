"""Tests for configuration management."""

import pytest
import os
from config import get_config, get_model_info, is_vision_model, RECOMMENDED_MODELS


def test_get_config_defaults():
    """Test default configuration values."""
    config = get_config()
    
    assert config.ollama_host == "http://localhost:11434"
    assert config.default_chat_model == "granite3.3:8b"
    assert config.default_vision_model == "llava:7b"
    assert config.request_timeout == 60
    assert config.max_file_size_mb == 10
    assert config.temperature == 0.7
    assert config.max_tokens == 2048


def test_get_config_environment_overrides():
    """Test configuration overrides from environment variables."""
    # Set environment variables
    os.environ["OLLAMA_HOST"] = "http://test:12345"
    os.environ["TEMPERATURE"] = "0.5"
    os.environ["MAX_TOKENS"] = "1024"
    
    config = get_config()
    
    assert config.ollama_host == "http://test:12345"
    assert config.temperature == 0.5
    assert config.max_tokens == 1024
    
    # Clean up
    del os.environ["OLLAMA_HOST"]
    del os.environ["TEMPERATURE"]
    del os.environ["MAX_TOKENS"]


def test_recommended_models():
    """Test recommended models configuration."""
    assert len(RECOMMENDED_MODELS) > 0
    
    # Check that all models have required fields
    for model in RECOMMENDED_MODELS:
        assert hasattr(model, 'name')
        assert hasattr(model, 'size')
        assert hasattr(model, 'type')
        assert hasattr(model, 'description')
        assert model.type in ['chat', 'vision', 'code']


def test_get_model_info():
    """Test getting model information."""
    # Test known model
    model_info = get_model_info("llava:7b")
    assert model_info.name == "llava:7b"
    assert model_info.type == "vision"
    
    # Test unknown model
    unknown_model = get_model_info("unknown:model")
    assert unknown_model.name == "unknown:model"
    assert unknown_model.description == "Model information not available"


def test_is_vision_model():
    """Test vision model detection."""
    # Test known vision models
    assert is_vision_model("llava:7b") == True
    assert is_vision_model("granite3.2-vision:latest") == True
    
    # Test chat models
    assert is_vision_model("granite3.3:8b") == False
    assert is_vision_model("gemma3:latest") == False
    
    # Test models with 'vision' in name
    assert is_vision_model("some-vision-model:latest") == True
    assert is_vision_model("llava-custom:7b") == True


def test_supported_image_formats():
    """Test supported image formats configuration."""
    config = get_config()
    
    expected_formats = ["jpg", "jpeg", "png", "webp", "gif"]
    assert all(fmt in config.supported_image_formats for fmt in expected_formats)