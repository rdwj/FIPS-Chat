"""Session management utilities for the Chat Playground application."""

import streamlit as st
from typing import Dict, Any, Optional
import json
from datetime import datetime, timedelta
from config import get_config


def initialize_session_state():
    """Initialize all session state variables."""
    default_values = {
        # Chat session
        "chat_messages": [],
        "chat_model": get_config().default_chat_model,
        "selected_chat_model": None,
        
        # Image session
        "uploaded_images": [],
        "image_analyses": [],
        "vision_model": get_config().default_vision_model,
        "selected_vision_model": None,
        
        # App state
        "current_tab": "Chat",
        "app_initialized": True,
        "session_start_time": datetime.now(),
        
        # Settings
        "temperature": get_config().temperature,
        "max_tokens": get_config().max_tokens,
        "theme": get_config().theme,
        
        # API Provider settings
        "api_provider": "ollama",
        "api_endpoint": "",
        "api_key": "",
        "api_type": "openai_compatible",
        "external_model_name": "",
        "agent_session_id": "",
        
        # Performance tracking
        "total_requests": 0,
        "total_response_time": 0,
        "error_count": 0,
    }
    
    # Initialize only if not already set
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_chat_session():
    """Clear chat-related session data."""
    st.session_state.chat_messages = []
    st.session_state.total_requests = 0
    st.session_state.total_response_time = 0


def clear_image_session():
    """Clear image-related session data."""
    st.session_state.uploaded_images = []
    st.session_state.image_analyses = []


def clear_all_session_data():
    """Clear all session data except app settings."""
    clear_chat_session()
    clear_image_session()
    st.session_state.error_count = 0


def get_session_statistics() -> Dict[str, Any]:
    """Get comprehensive session statistics."""
    stats = {
        "session_duration": get_session_duration(),
        "chat_stats": get_chat_stats(),
        "image_stats": get_image_stats(),
        "performance_stats": get_performance_stats(),
        "error_stats": get_error_stats(),
    }
    return stats


def get_session_duration() -> Dict[str, Any]:
    """Get session duration information."""
    if "session_start_time" not in st.session_state:
        st.session_state.session_start_time = datetime.now()
    
    duration = datetime.now() - st.session_state.session_start_time
    
    return {
        "start_time": st.session_state.session_start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": duration.total_seconds(),
        "duration_formatted": format_duration(duration),
    }


def get_chat_stats() -> Dict[str, Any]:
    """Get chat session statistics."""
    messages = st.session_state.get("chat_messages", [])
    
    user_messages = [msg for msg in messages if msg.get("role") == "user"]
    assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
    
    # Calculate response times
    response_times = [msg.get("response_time", 0) for msg in assistant_messages if msg.get("response_time", 0) > 0]
    
    return {
        "total_messages": len(messages),
        "user_messages": len(user_messages),
        "assistant_messages": len(assistant_messages),
        "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
        "total_response_time": sum(response_times),
        "fastest_response": min(response_times) if response_times else 0,
        "slowest_response": max(response_times) if response_times else 0,
    }


def get_image_stats() -> Dict[str, Any]:
    """Get image session statistics."""
    images = st.session_state.get("uploaded_images", [])
    analyses = st.session_state.get("image_analyses", [])
    
    # Calculate analysis response times
    response_times = [analysis.get("response_time", 0) for analysis in analyses if analysis.get("response_time", 0) > 0]
    
    # Calculate total file size
    total_size = sum(img.get("size", 0) for img in images)
    
    return {
        "total_images": len(images),
        "total_analyses": len(analyses),
        "total_file_size": total_size,
        "avg_analysis_time": sum(response_times) / len(response_times) if response_times else 0,
        "total_analysis_time": sum(response_times),
    }


def get_performance_stats() -> Dict[str, Any]:
    """Get overall performance statistics."""
    total_requests = st.session_state.get("total_requests", 0)
    total_response_time = st.session_state.get("total_response_time", 0)
    
    return {
        "total_requests": total_requests,
        "total_response_time": total_response_time,
        "avg_response_time": total_response_time / total_requests if total_requests > 0 else 0,
    }


def get_error_stats() -> Dict[str, Any]:
    """Get error statistics."""
    return {
        "total_errors": st.session_state.get("error_count", 0),
    }


def track_request_performance(response_time: float):
    """Track request performance metrics."""
    st.session_state.total_requests = st.session_state.get("total_requests", 0) + 1
    st.session_state.total_response_time = st.session_state.get("total_response_time", 0) + response_time


def track_error():
    """Track error occurrence."""
    st.session_state.error_count = st.session_state.get("error_count", 0) + 1


def format_duration(duration: timedelta) -> str:
    """Format duration in human-readable format."""
    total_seconds = int(duration.total_seconds())
    
    if total_seconds < 60:
        return f"{total_seconds}s"
    elif total_seconds < 3600:
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes}m {seconds}s"
    else:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"{hours}h {minutes}m"


def export_session_data() -> str:
    """Export session data as JSON string."""
    export_data = {
        "export_timestamp": datetime.now().isoformat(),
        "session_stats": get_session_statistics(),
        "chat_messages": st.session_state.get("chat_messages", []),
        "image_analyses": st.session_state.get("image_analyses", []),
        "settings": {
            "chat_model": st.session_state.get("chat_model"),
            "vision_model": st.session_state.get("vision_model"),
            "temperature": st.session_state.get("temperature"),
            "max_tokens": st.session_state.get("max_tokens"),
        }
    }
    
    return json.dumps(export_data, indent=2, default=str)


def get_memory_usage_estimate() -> Dict[str, Any]:
    """Estimate memory usage of session data."""
    chat_size = len(json.dumps(st.session_state.get("chat_messages", []), default=str))
    image_size = sum(len(img.get("data", b"")) for img in st.session_state.get("uploaded_images", []))
    analysis_size = len(json.dumps(st.session_state.get("image_analyses", []), default=str))
    
    total_size = chat_size + image_size + analysis_size
    
    return {
        "chat_data_bytes": chat_size,
        "image_data_bytes": image_size,
        "analysis_data_bytes": analysis_size,
        "total_bytes": total_size,
        "total_mb": total_size / (1024 * 1024),
        "memory_warning": total_size > 50 * 1024 * 1024,  # Warn if > 50MB
    }


def cleanup_old_data(max_messages: int = 100, max_images: int = 10):
    """Clean up old data to prevent memory issues."""
    # Limit chat messages
    if len(st.session_state.get("chat_messages", [])) > max_messages:
        st.session_state.chat_messages = st.session_state.chat_messages[-max_messages:]
    
    # Limit uploaded images
    if len(st.session_state.get("uploaded_images", [])) > max_images:
        st.session_state.uploaded_images = st.session_state.uploaded_images[-max_images:]
        # Also clean up corresponding analyses
        recent_image_names = {img["name"] for img in st.session_state.uploaded_images}
        st.session_state.image_analyses = [
            analysis for analysis in st.session_state.get("image_analyses", [])
            if analysis.get("image_name") in recent_image_names
        ]


def should_show_memory_warning() -> bool:
    """Check if memory usage warning should be displayed."""
    memory_info = get_memory_usage_estimate()
    return memory_info["memory_warning"]