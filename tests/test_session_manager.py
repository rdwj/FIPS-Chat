"""Tests for session management utilities."""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from utils.session_manager import (
    get_session_duration,
    get_chat_stats,
    get_image_stats,
    format_duration,
    get_memory_usage_estimate,
    should_show_memory_warning
)


@pytest.fixture
def mock_session_state():
    """Mock streamlit session state."""
    return {
        "session_start_time": datetime.now() - timedelta(minutes=30),
        "chat_messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!", "response_time": 1.5},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well!", "response_time": 2.0}
        ],
        "uploaded_images": [
            {"name": "test1.jpg", "size": 1024000, "data": b"x" * 1000},
            {"name": "test2.png", "size": 2048000, "data": b"y" * 2000}
        ],
        "image_analyses": [
            {"image_name": "test1.jpg", "response_time": 3.0},
            {"image_name": "test2.png", "response_time": 4.5},
            {"image_name": "test1.jpg", "response_time": 2.5}
        ]
    }


def test_get_session_duration(mock_session_state):
    """Test session duration calculation."""
    # Create a mock class to simulate streamlit.session_state behavior
    class MockSessionState:
        def __init__(self, data):
            self._data = data
        
        def __contains__(self, key):
            return key in self._data
        
        def __getitem__(self, key):
            return self._data[key]
        
        def __setitem__(self, key, value):
            self._data[key] = value
        
        @property
        def session_start_time(self):
            return self._data.get("session_start_time")
    
    mock_state = MockSessionState(mock_session_state)
    
    with patch('utils.session_manager.st.session_state', mock_state):
        duration_info = get_session_duration()
        
        assert "start_time" in duration_info
        assert "duration_seconds" in duration_info
        assert "duration_formatted" in duration_info
        
        # Should be approximately 30 minutes (1800 seconds)
        assert duration_info["duration_seconds"] >= 1750
        assert duration_info["duration_seconds"] <= 1850


def test_get_chat_stats(mock_session_state):
    """Test chat statistics calculation."""
    with patch('streamlit.session_state', mock_session_state):
        chat_stats = get_chat_stats()
        
        assert chat_stats["total_messages"] == 4
        assert chat_stats["user_messages"] == 2
        assert chat_stats["assistant_messages"] == 2
        assert chat_stats["avg_response_time"] == 1.75  # (1.5 + 2.0) / 2
        assert chat_stats["total_response_time"] == 3.5
        assert chat_stats["fastest_response"] == 1.5
        assert chat_stats["slowest_response"] == 2.0


def test_get_image_stats(mock_session_state):
    """Test image statistics calculation."""
    with patch('streamlit.session_state', mock_session_state):
        image_stats = get_image_stats()
        
        assert image_stats["total_images"] == 2
        assert image_stats["total_analyses"] == 3
        assert image_stats["total_file_size"] == 3072000  # 1024000 + 2048000
        assert image_stats["avg_analysis_time"] == 10.0 / 3  # (3.0 + 4.5 + 2.5) / 3
        assert image_stats["total_analysis_time"] == 10.0


def test_format_duration():
    """Test duration formatting."""
    assert format_duration(timedelta(seconds=30)) == "30s"
    assert format_duration(timedelta(minutes=2, seconds=30)) == "2m 30s"
    assert format_duration(timedelta(hours=1, minutes=30)) == "1h 30m"
    assert format_duration(timedelta(hours=2, minutes=0, seconds=45)) == "2h 0m"


def test_get_memory_usage_estimate(mock_session_state):
    """Test memory usage estimation."""
    with patch('streamlit.session_state', mock_session_state):
        memory_info = get_memory_usage_estimate()
        
        assert "chat_data_bytes" in memory_info
        assert "image_data_bytes" in memory_info
        assert "analysis_data_bytes" in memory_info
        assert "total_bytes" in memory_info
        assert "total_mb" in memory_info
        assert "memory_warning" in memory_info
        
        # Image data should be the largest component
        assert memory_info["image_data_bytes"] == 3000  # 1000 + 2000
        assert memory_info["total_bytes"] > 0


def test_should_show_memory_warning():
    """Test memory warning logic."""
    # Test with small memory usage
    small_session = {
        "chat_messages": [{"role": "user", "content": "Hi"}],
        "uploaded_images": [],
        "image_analyses": []
    }
    
    with patch('streamlit.session_state', small_session):
        assert should_show_memory_warning() == False
    
    # Test with large memory usage
    large_data = b"x" * (60 * 1024 * 1024)  # 60MB
    large_session = {
        "chat_messages": [],
        "uploaded_images": [{"data": large_data}],
        "image_analyses": []
    }
    
    with patch('streamlit.session_state', large_session):
        assert should_show_memory_warning() == True


def test_get_chat_stats_empty():
    """Test chat stats with empty messages."""
    empty_session = {"chat_messages": []}
    
    with patch('streamlit.session_state', empty_session):
        chat_stats = get_chat_stats()
        
        assert chat_stats["total_messages"] == 0
        assert chat_stats["user_messages"] == 0
        assert chat_stats["assistant_messages"] == 0
        assert chat_stats["avg_response_time"] == 0


def test_get_image_stats_empty():
    """Test image stats with no images."""
    empty_session = {
        "uploaded_images": [],
        "image_analyses": []
    }
    
    with patch('streamlit.session_state', empty_session):
        image_stats = get_image_stats()
        
        assert image_stats["total_images"] == 0
        assert image_stats["total_analyses"] == 0
        assert image_stats["total_file_size"] == 0
        assert image_stats["avg_analysis_time"] == 0


def test_chat_stats_no_response_times():
    """Test chat stats when no response times are available."""
    session_no_times = {
        "chat_messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"}  # No response_time
        ]
    }
    
    with patch('streamlit.session_state', session_no_times):
        chat_stats = get_chat_stats()
        
        assert chat_stats["avg_response_time"] == 0
        assert chat_stats["total_response_time"] == 0
        assert chat_stats["fastest_response"] == 0
        assert chat_stats["slowest_response"] == 0


def test_image_stats_no_response_times():
    """Test image stats when no response times are available."""
    session_no_times = {
        "uploaded_images": [{"name": "test.jpg", "size": 1000}],
        "image_analyses": [{"image_name": "test.jpg"}]  # No response_time
    }
    
    with patch('streamlit.session_state', session_no_times):
        image_stats = get_image_stats()
        
        assert image_stats["avg_analysis_time"] == 0
        assert image_stats["total_analysis_time"] == 0