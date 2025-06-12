"""Tests for Ollama client functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from ollama_client import OllamaClient, ModelInfo
from config import AppConfig
import requests


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = AppConfig()
    config.ollama_host = "http://localhost:11434"
    config.request_timeout = 30
    config.temperature = 0.7
    config.max_tokens = 2048
    return config


@pytest.fixture
def ollama_client(mock_config):
    """Create an Ollama client for testing."""
    return OllamaClient(mock_config)


def test_ollama_client_initialization(mock_config):
    """Test Ollama client initialization."""
    client = OllamaClient(mock_config)
    
    assert client.config == mock_config
    assert client.base_url == "http://localhost:11434"
    assert client.session.timeout == 30


@patch('requests.Session.get')
def test_get_available_models_success(mock_get, ollama_client):
    """Test successful retrieval of available models."""
    # Mock response
    mock_response = Mock()
    mock_response.json.return_value = {
        "models": [
            {
                "name": "llava:7b",
                "size": 5000000000,
                "digest": "abc123",
                "modified_at": "2024-01-01T00:00:00Z",
                "details": {"family": "llava"}
            },
            {
                "name": "granite3.3:8b",
                "size": 5200000000,
                "digest": "def456",
                "modified_at": "2024-01-01T00:00:00Z",
                "details": {"family": "granite"}
            }
        ]
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    
    models = ollama_client.get_available_models()
    
    assert len(models) == 2
    assert models[0].name == "llava:7b"
    assert models[0].size == 5000000000
    assert models[1].name == "granite3.3:8b"


@patch('requests.Session.get')
def test_get_available_models_connection_error(mock_get, ollama_client):
    """Test handling of connection error when getting models."""
    mock_get.side_effect = requests.exceptions.ConnectionError()
    
    models = ollama_client.get_available_models()
    
    assert models == []


@patch('requests.Session.post')
def test_get_model_info_success(mock_post, ollama_client):
    """Test successful model info retrieval."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "modelfile": "FROM llava:7b",
        "parameters": {"temperature": 0.7},
        "template": "{{ .System }}\n{{ .Prompt }}"
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response
    
    info = ollama_client.get_model_info("llava:7b")
    
    assert info is not None
    assert "modelfile" in info
    mock_post.assert_called_once()


def test_is_model_available(ollama_client):
    """Test model availability check."""
    with patch.object(ollama_client, 'get_available_models') as mock_models:
        mock_models.return_value = [
            ModelInfo("llava:7b", 5000000000, "abc123", "2024-01-01T00:00:00Z", {}),
            ModelInfo("granite3.3:8b", 5200000000, "def456", "2024-01-01T00:00:00Z", {})
        ]
        
        assert ollama_client.is_model_available("llava:7b") == True
        assert ollama_client.is_model_available("unknown:model") == False


@patch('requests.Session.post')
def test_chat_streaming(mock_post, ollama_client):
    """Test chat with streaming response."""
    # Mock streaming response
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.iter_lines.return_value = [
        b'{"message": {"content": "Hello"}, "done": false}',
        b'{"message": {"content": " there"}, "done": false}',
        b'{"message": {"content": "!"}, "done": true}'
    ]
    mock_post.return_value = mock_response
    
    messages = [{"role": "user", "content": "Hi"}]
    response_chunks = list(ollama_client.chat("llava:7b", messages, stream=True))
    
    assert response_chunks == ["Hello", " there", "!"]


@patch('requests.Session.post')
def test_chat_connection_error(mock_post, ollama_client):
    """Test chat with connection error."""
    mock_post.side_effect = requests.exceptions.ConnectionError()
    
    messages = [{"role": "user", "content": "Hi"}]
    response_chunks = list(ollama_client.chat("llava:7b", messages))
    
    assert len(response_chunks) == 1
    assert "Could not connect to Ollama" in response_chunks[0]


@patch('requests.Session.post')
def test_generate_with_image_success(mock_post, ollama_client):
    """Test image generation with success."""
    # Mock streaming response
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.iter_lines.return_value = [
        b'{"response": "This image shows", "done": false}',
        b'{"response": " a beautiful landscape", "done": true}'
    ]
    mock_post.return_value = mock_response
    
    image_data = b"fake_image_data"
    response_chunks = list(ollama_client.generate_with_image(
        "llava:7b", "Describe this image", image_data, stream=True
    ))
    
    assert response_chunks == ["This image shows", " a beautiful landscape"]


def test_error_handling_timeout(ollama_client):
    """Test timeout error handling."""
    with patch('requests.Session.post') as mock_post:
        mock_post.side_effect = requests.exceptions.Timeout()
        
        messages = [{"role": "user", "content": "Hi"}]
        response_chunks = list(ollama_client.chat("llava:7b", messages))
        
        assert len(response_chunks) == 1
        assert "timed out" in response_chunks[0].lower()


def test_error_handling_http_error(ollama_client):
    """Test HTTP error handling."""
    with patch('requests.Session.post') as mock_post:
        mock_response = Mock()
        mock_error = requests.exceptions.HTTPError()
        mock_error.response = Mock()
        mock_error.response.status_code = 404
        mock_error.response.text = "Not Found"
        mock_response.raise_for_status.side_effect = mock_error
        mock_post.return_value = mock_response
        
        with pytest.raises(ValueError):
            ollama_client._make_request("nonexistent", "POST", {})