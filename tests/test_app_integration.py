"""Integration tests for the main application."""

import pytest
import sys
import os
from unittest.mock import patch, Mock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app


def test_app_imports():
    """Test that all app modules can be imported without errors."""
    # Test importing core modules
    import config
    import ollama_client
    from ui_components import model_selector, chat_interface, image_interface
    from utils import image_processing, session_manager
    
    # Verify key functions exist
    assert hasattr(config, 'get_config')
    assert hasattr(ollama_client, 'OllamaClient')
    assert hasattr(model_selector, 'render_model_selector')
    assert hasattr(chat_interface, 'render_chat_interface')
    assert hasattr(image_interface, 'render_image_interface')


@patch('streamlit.set_page_config')
@patch('app.render_header')
@patch('app.render_sidebar')
@patch('app.render_main_content')
@patch('app.handle_performance_monitoring')
@patch('utils.session_manager.initialize_session_state')
def test_main_function_calls(mock_init, mock_perf, mock_main, mock_sidebar, mock_header, mock_config):
    """Test that main function calls all required components."""
    app.main()
    
    # Verify all main components are called
    mock_config.assert_called_once()
    mock_init.assert_called_once()
    mock_header.assert_called_once()
    mock_sidebar.assert_called_once()
    mock_main.assert_called_once()
    mock_perf.assert_called_once()


def test_config_values():
    """Test that configuration values are reasonable."""
    from config import get_config, RECOMMENDED_MODELS
    
    config = get_config()
    
    # Test basic config values
    assert config.ollama_host.startswith('http')
    assert config.request_timeout > 0
    assert config.max_file_size_mb > 0
    assert len(config.supported_image_formats) > 0
    
    # Test recommended models
    assert len(RECOMMENDED_MODELS) > 0
    vision_models = [m for m in RECOMMENDED_MODELS if m.type == 'vision']
    chat_models = [m for m in RECOMMENDED_MODELS if m.type == 'chat']
    
    assert len(vision_models) > 0
    assert len(chat_models) > 0


@patch('ollama_client.requests.Session')
def test_ollama_client_creation(mock_session):
    """Test that Ollama client can be created."""
    from ollama_client import OllamaClient
    from config import get_config
    
    config = get_config()
    client = OllamaClient(config)
    
    assert client.config == config
    assert client.base_url == config.ollama_host


def test_image_processing_utilities():
    """Test basic image processing functionality."""
    from utils.image_processing import format_file_size, get_supported_formats_info
    
    # Test file size formatting
    assert format_file_size(1024) == "1.0 KB"
    assert format_file_size(1048576) == "1.0 MB"
    
    # Test supported formats
    formats = get_supported_formats_info()
    assert isinstance(formats, dict)
    assert 'jpg' in formats
    assert 'png' in formats


def test_session_manager_utilities():
    """Test session management utilities."""
    from utils.session_manager import format_duration
    from datetime import timedelta
    
    # Test duration formatting
    assert format_duration(timedelta(seconds=30)) == "30s"
    assert format_duration(timedelta(minutes=5)) == "5m 0s"


@patch('streamlit.session_state', {})
def test_app_functions_exist():
    """Test that all main app functions exist and can be called."""
    # These functions should exist and be callable
    assert hasattr(app, 'main')
    assert hasattr(app, 'render_header')
    assert hasattr(app, 'render_sidebar')
    assert hasattr(app, 'render_main_content')
    assert hasattr(app, 'check_ollama_connection')
    
    # These should be callable (though they might raise exceptions without proper mocking)
    assert callable(app.main)
    assert callable(app.render_header)
    assert callable(app.render_sidebar)