"""Session management utilities for the FIPS Chat application."""

import streamlit as st
from typing import Dict, Any, Optional
import json
from datetime import datetime, timedelta
from config import get_config, RAGConfig, RAGConfigManager


def initialize_session_state():
    """Initialize all session state variables."""
    config = get_config()
    
    default_values = {
        # Chat session
        "chat_messages": [],
        "selected_chat_model": None,
        
        # Image session
        "uploaded_images": [],
        "image_analyses": [],
        "selected_vision_model": None,
        
        # App state
        "current_tab": "Chat",
        "app_initialized": True,
        "session_start_time": datetime.now(),
        
        # Settings
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "theme": config.theme,
        
        # API settings
        "api_provider": config.default_api_provider,
        "api_endpoint": config.default_api_endpoint or "",
        "api_key": config.default_api_key or "",
        "agent_session_id": "",
        
        # Model discovery
        "discovered_models": [],
        "auto_discover_models": config.auto_discover_models,
        
        # Performance tracking
        "total_requests": 0,
        "total_response_time": 0,
        "error_count": 0,
    }
    
    # Initialize only if not already set
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Initialize RAG-specific session state
    initialize_rag_session_state(config.rag)


def initialize_rag_session_state(rag_config: RAGConfig):
    """Initialize RAG-specific session state"""
    rag_defaults = {
        # RAG configuration
        "rag_config": rag_config,
        "rag_config_manager": RAGConfigManager(),
        
        # RAG state
        "rag_enabled": rag_config.enabled,
        "rag_max_sources": rag_config.max_search_results,
        "rag_threshold": rag_config.relevance_threshold,
        
        # Document management
        "uploaded_documents": [],
        "processed_documents": [],
        "document_processing_status": {},
        "document_errors": [],
        
        # RAG storage and pipeline
        "rag_storage_initialized": False,
        "rag_pipeline": None,
        "document_count": 0,
        "index_last_updated": None,
        
        # Search and retrieval
        "rag_search_history": [],
        "last_search_results": [],
        "context_sources": [],
        
        # Performance tracking
        "rag_requests": 0,
        "rag_response_time": 0,
        "document_processing_time": 0,
        "search_time": 0,
        
        # Configuration state
        "rag_config_dirty": False,
        "rag_config_validation_errors": [],
        
        # Demo mode tracking
        "demo_pages_processed": 0,
        "demo_pdfs_uploaded": 0,
        "demo_limits_reached": False,
    }
    
    # Initialize only if not already set
    for key, value in rag_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def update_rag_config(new_config: RAGConfig):
    """Update RAG configuration in session state"""
    st.session_state.rag_config = new_config
    st.session_state.rag_enabled = new_config.enabled
    st.session_state.rag_max_sources = new_config.max_search_results
    st.session_state.rag_threshold = new_config.relevance_threshold
    st.session_state.rag_config_dirty = True


def validate_rag_session_config() -> bool:
    """Validate current RAG configuration and update session state"""
    if "rag_config_manager" not in st.session_state:
        return False
    
    try:
        config_manager = st.session_state.rag_config_manager
        errors = config_manager.validator.validate_config(st.session_state.rag_config)
        st.session_state.rag_config_validation_errors = errors
        return len(errors) == 0
    except Exception as e:
        st.session_state.rag_config_validation_errors = [str(e)]
        return False


def clear_rag_session():
    """Clear RAG-related session data"""
    rag_keys_to_clear = [
        "uploaded_documents",
        "processed_documents", 
        "document_processing_status",
        "document_errors",
        "rag_search_history",
        "last_search_results",
        "context_sources",
        "demo_pages_processed",
        "demo_pdfs_uploaded"
    ]
    
    for key in rag_keys_to_clear:
        if key in st.session_state:
            st.session_state[key] = []
    
    # Reset counters
    st.session_state.document_count = 0
    st.session_state.rag_requests = 0
    st.session_state.rag_response_time = 0
    st.session_state.document_processing_time = 0
    st.session_state.search_time = 0
    st.session_state.demo_limits_reached = False
    st.session_state.rag_storage_initialized = False
    st.session_state.index_last_updated = None


def check_demo_limits() -> Dict[str, Any]:
    """Check if demo limits have been reached"""
    if not st.session_state.get("rag_config", RAGConfig()).demo_mode:
        return {"limit_reached": False, "warnings": []}
    
    config = st.session_state.rag_config
    warnings = []
    
    # Check page limit
    pages_processed = st.session_state.get("demo_pages_processed", 0)
    if pages_processed >= config.demo_max_pages:
        warnings.append(f"Demo page limit reached ({config.demo_max_pages} pages)")
    elif pages_processed > config.demo_max_pages * 0.8:
        warnings.append(f"Approaching demo page limit ({pages_processed}/{config.demo_max_pages} pages)")
    
    # Check PDF limit
    pdfs_uploaded = st.session_state.get("demo_pdfs_uploaded", 0)
    if pdfs_uploaded >= config.demo_max_pdfs:
        warnings.append(f"Demo PDF limit reached ({config.demo_max_pdfs} PDFs)")
    elif pdfs_uploaded > config.demo_max_pdfs * 0.8:
        warnings.append(f"Approaching demo PDF limit ({pdfs_uploaded}/{config.demo_max_pdfs} PDFs)")
    
    limit_reached = any("limit reached" in warning for warning in warnings)
    st.session_state.demo_limits_reached = limit_reached
    
    return {
        "limit_reached": limit_reached,
        "warnings": warnings,
        "pages_processed": pages_processed,
        "pdfs_uploaded": pdfs_uploaded,
        "max_pages": config.demo_max_pages,
        "max_pdfs": config.demo_max_pdfs
    }


def track_rag_performance(operation: str, response_time: float):
    """Track RAG operation performance"""
    st.session_state.rag_requests = st.session_state.get("rag_requests", 0) + 1
    st.session_state.rag_response_time = st.session_state.get("rag_response_time", 0) + response_time
    
    if operation == "document_processing":
        st.session_state.document_processing_time = st.session_state.get("document_processing_time", 0) + response_time
    elif operation == "search":
        st.session_state.search_time = st.session_state.get("search_time", 0) + response_time


def get_rag_stats() -> Dict[str, Any]:
    """Get RAG session statistics"""
    config = st.session_state.get("rag_config", RAGConfig())
    
    return {
        "enabled": st.session_state.get("rag_enabled", False),
        "documents_uploaded": len(st.session_state.get("uploaded_documents", [])),
        "documents_processed": len(st.session_state.get("processed_documents", [])),
        "total_rag_requests": st.session_state.get("rag_requests", 0),
        "avg_rag_response_time": (
            st.session_state.get("rag_response_time", 0) / 
            max(1, st.session_state.get("rag_requests", 1))
        ),
        "document_processing_time": st.session_state.get("document_processing_time", 0),
        "search_time": st.session_state.get("search_time", 0),
        "search_history_count": len(st.session_state.get("rag_search_history", [])),
        "storage_initialized": st.session_state.get("rag_storage_initialized", False),
        "index_last_updated": st.session_state.get("index_last_updated"),
        "demo_mode": config.demo_mode,
        "demo_pages_processed": st.session_state.get("demo_pages_processed", 0),
        "demo_pdfs_uploaded": st.session_state.get("demo_pdfs_uploaded", 0),
        "demo_limits_reached": st.session_state.get("demo_limits_reached", False),
        "config_validation_errors": st.session_state.get("rag_config_validation_errors", [])
    }


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
    clear_rag_session()
    st.session_state.error_count = 0


def get_session_statistics() -> Dict[str, Any]:
    """Get comprehensive session statistics."""
    stats = {
        "session_duration": get_session_duration(),
        "chat_stats": get_chat_stats(),
        "image_stats": get_image_stats(),
        "rag_stats": get_rag_stats(),
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
        "rag_data": {
            "uploaded_documents": st.session_state.get("uploaded_documents", []),
            "processed_documents": st.session_state.get("processed_documents", []),
            "search_history": st.session_state.get("rag_search_history", []),
            "document_errors": st.session_state.get("document_errors", []),
        },
        "settings": {
            "selected_chat_model": st.session_state.get("selected_chat_model"),
            "selected_vision_model": st.session_state.get("selected_vision_model"),
            "api_endpoint": st.session_state.get("api_endpoint"),
            "api_provider": st.session_state.get("api_provider"),
            "temperature": st.session_state.get("temperature"),
            "max_tokens": st.session_state.get("max_tokens"),
            "rag_enabled": st.session_state.get("rag_enabled", False),
            "rag_max_sources": st.session_state.get("rag_max_sources", 5),
            "rag_threshold": st.session_state.get("rag_threshold", 0.1),
        }
    }
    
    return json.dumps(export_data, indent=2, default=str)


def get_memory_usage_estimate() -> Dict[str, Any]:
    """Estimate memory usage of session data."""
    chat_size = len(json.dumps(st.session_state.get("chat_messages", []), default=str))
    image_size = sum(len(img.get("data", b"")) for img in st.session_state.get("uploaded_images", []))
    analysis_size = len(json.dumps(st.session_state.get("image_analyses", []), default=str))
    
    # RAG data size estimation
    rag_docs_size = len(json.dumps(st.session_state.get("uploaded_documents", []), default=str))
    rag_processed_size = len(json.dumps(st.session_state.get("processed_documents", []), default=str))
    rag_search_size = len(json.dumps(st.session_state.get("rag_search_history", []), default=str))
    rag_total_size = rag_docs_size + rag_processed_size + rag_search_size
    
    total_size = chat_size + image_size + analysis_size + rag_total_size
    
    # Get configured memory limit for RAG
    rag_config = st.session_state.get("rag_config", RAGConfig())
    rag_limit_bytes = rag_config.max_memory_mb * 1024 * 1024
    
    return {
        "chat_data_bytes": chat_size,
        "image_data_bytes": image_size,
        "analysis_data_bytes": analysis_size,
        "rag_data_bytes": rag_total_size,
        "total_bytes": total_size,
        "total_mb": total_size / (1024 * 1024),
        "rag_limit_mb": rag_config.max_memory_mb,
        "rag_usage_percent": (rag_total_size / rag_limit_bytes * 100) if rag_limit_bytes > 0 else 0,
        "memory_warning": total_size > 50 * 1024 * 1024,  # Warn if > 50MB
        "rag_memory_warning": rag_total_size > rag_limit_bytes * 0.8,  # Warn if RAG uses > 80% of limit
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
    
    # RAG-specific cleanup based on configuration
    rag_config = st.session_state.get("rag_config", RAGConfig())
    
    # Limit search history
    max_search_history = 50  # Keep last 50 searches
    if len(st.session_state.get("rag_search_history", [])) > max_search_history:
        st.session_state.rag_search_history = st.session_state.rag_search_history[-max_search_history:]
    
    # Limit document processing errors (keep last 20)
    if len(st.session_state.get("document_errors", [])) > 20:
        st.session_state.document_errors = st.session_state.document_errors[-20:]
    
    # Check memory usage and clean up if needed
    memory_info = get_memory_usage_estimate()
    if memory_info.get("rag_memory_warning", False):
        # Remove oldest processed documents if memory usage is high
        processed_docs = st.session_state.get("processed_documents", [])
        if len(processed_docs) > 10:
            st.session_state.processed_documents = processed_docs[-10:]  # Keep only last 10


def should_show_memory_warning() -> bool:
    """Check if memory usage warning should be displayed."""
    memory_info = get_memory_usage_estimate()
    return memory_info["memory_warning"]