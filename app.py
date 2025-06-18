"""FIPS Chat - Multi-provider AI chat and image analysis platform."""

import streamlit as st
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config, get_all_api_templates, get_api_template
from ui_components.model_selector import render_model_selector, render_model_management
from ui_components.chat_interface import render_chat_interface, get_chat_statistics
from ui_components.image_interface import render_image_interface, get_image_statistics
from ui_components.document_interface import render_document_interface, get_document_statistics
from ui_components.rag_chat_interface import render_rag_chat_interface, get_rag_chat_statistics
from utils.session_manager import (
    initialize_session_state, 
    get_session_statistics, 
    clear_all_session_data,
    should_show_memory_warning,
    cleanup_old_data
)
from ai_client import get_ai_client, test_api_connection, ModelCapability


def main():
    """Main application function."""
    # Page configuration
    st.set_page_config(
        page_title="FIPS Chat",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    render_header()
    
    # Sidebar
    render_sidebar()
    
    # Main content area
    render_main_content()
    
    # Performance monitoring
    handle_performance_monitoring()


def render_header():
    """Render application header."""
    st.title("💬 FIPS Chat")
    st.markdown("*FIPS-compliant multi-provider AI chat and image analysis platform*")
    
    # Connection status
    check_api_connection()


def check_api_connection():
    """Check and display API connection status."""
    api_endpoint = st.session_state.get("api_endpoint")
    
    if not api_endpoint:
        st.warning("⚠️ Please configure your API endpoint")
        return
    
    try:
        client = get_ai_client()
        if client:
            is_connected, message = client.test_connection()
            if is_connected:
                # Try to discover models to get count
                try:
                    models = client.discover_models()
                    if models:
                        st.success(f"✅ {message} • {len(models)} models discovered")
                    else:
                        st.success(f"✅ {message}")
                except Exception:
                    st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")
        else:
            st.error("❌ Could not create API client")
    except Exception as e:
        st.error(f"❌ API connection error: {str(e)}")


def render_sidebar():
    """Render sidebar with controls and information."""
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # API configuration
        render_api_configuration()
        
        st.divider()
        
        # Model discovery and selection
        render_model_discovery()
        
        st.divider()
        
        # Settings
        render_settings()
        
        st.divider()
        
        # Statistics
        render_statistics()
        
        st.divider()
        
        # Actions
        render_sidebar_actions()


def render_api_configuration():
    """Render API endpoint configuration."""
    st.subheader("🌐 API Configuration")
    
    # API template selection
    templates = get_all_api_templates()
    template_options = ["custom"] + list(templates.keys())
    
    selected_template = st.selectbox(
        "API Template",
        template_options,
        index=0,
        help="Choose a pre-configured template or use custom settings"
    )
    
    # Load template if selected
    if selected_template != "custom":
        template = get_api_template(selected_template)
        if template:
            st.info(f"**{template['name']}** - {template['endpoint']}")
            if template.get('example_models'):
                st.caption(f"Example models: {', '.join(template['example_models'][:3])}")
    
    # API endpoint configuration
    default_endpoint = ""
    if selected_template != "custom":
        template = get_api_template(selected_template)
        if template:
            default_endpoint = template["endpoint"]
    
    api_endpoint = st.text_input(
        "API Endpoint",
        value=st.session_state.get("api_endpoint", default_endpoint),
        placeholder="https://api.example.com/v1 or http://localhost:8000/v1",
        help="Full URL to your API endpoint"
    )
    st.session_state.api_endpoint = api_endpoint
    
    # API key (conditional based on template)
    requires_key = True
    if selected_template != "custom":
        template = get_api_template(selected_template)
        if template:
            requires_key = template.get("requires_key", True)
    
    if requires_key:
        api_key = st.text_input(
            "API Key",
            value=st.session_state.get("api_key", ""),
            type="password",
            placeholder="Your API key",
            help="API authentication key"
        )
    else:
        api_key = st.text_input(
            "API Key (Optional)",
            value=st.session_state.get("api_key", ""),
            type="password",
            placeholder="Leave empty if not required",
            help="API authentication key (if required)"
        )
    
    st.session_state.api_key = api_key
    
    # Store provider type for backward compatibility
    if selected_template in ["openai", "vllm", "ollama", "text-generation-webui"]:
        st.session_state.api_provider = "openai_compatible"
    elif selected_template == "anthropic":
        st.session_state.api_provider = "anthropic"
    else:
        st.session_state.api_provider = "openai_compatible"  # Default
    
    # Test connection button
    if api_endpoint:
        if st.button("🔍 Test Connection"):
            test_api_connection_ui()


def test_api_connection_ui():
    """Test API connection and show results in UI."""
    api_endpoint = st.session_state.get("api_endpoint")
    api_key = st.session_state.get("api_key")
    provider = st.session_state.get("api_provider", "openai_compatible")
    
    if not api_endpoint:
        st.error("Please enter an API endpoint")
        return
    
    try:
        with st.spinner("Testing connection..."):
            is_connected, message = test_api_connection(api_endpoint, api_key, provider)
            
        if is_connected:
            st.success(f"✅ {message}")
            # Try to discover models
            try:
                client = get_ai_client()
                if client:
                    models = client.discover_models()
                    if models:
                        st.info(f"🔍 Discovered {len(models)} models")
                        # Show first few models
                        model_names = [m.name for m in models[:5]]
                        st.caption(f"Examples: {', '.join(model_names)}{'...' if len(models) > 5 else ''}")
            except Exception as e:
                st.warning(f"Model discovery failed: {str(e)}")
        else:
            st.error(f"❌ {message}")
            
    except Exception as e:
        st.error(f"❌ Error testing connection: {str(e)}")


def render_model_discovery():
    """Render model discovery and selection interface."""
    st.subheader("🤖 Model Selection")
    
    client = get_ai_client()
    if not client:
        st.warning("Please configure your API endpoint first")
        return
    
    # Model discovery
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🔍 Discover Models", help="Refresh model list from API"):
            with st.spinner("Discovering models..."):
                try:
                    models = client.discover_models(force_refresh=True)
                    st.session_state["discovered_models"] = models
                    if models:
                        st.success(f"✅ Found {len(models)} models")
                    else:
                        st.warning("No models found")
                except Exception as e:
                    st.error(f"Model discovery failed: {str(e)}")
    
    with col2:
        auto_discover = st.checkbox(
            "Auto-discover",
            value=st.session_state.get("auto_discover_models", True),
            help="Automatically discover models on connection"
        )
        st.session_state.auto_discover_models = auto_discover
    
    # Show discovered models
    models = st.session_state.get("discovered_models", [])
    
    if models:
        # Filter models by capability
        chat_models = [m for m in models if ModelCapability.CHAT in m.capabilities]
        vision_models = [m for m in models if ModelCapability.VISION in m.capabilities]
        
        if chat_models:
            st.write("**💬 Chat Models:**")
            selected_chat = st.selectbox(
                "Select Chat Model",
                [m.id for m in chat_models],
                index=0,
                key="selected_chat_model_select"
            )
            st.session_state.selected_chat_model = selected_chat
        
        if vision_models:
            st.write("**🖼️ Vision Models:**")
            selected_vision = st.selectbox(
                "Select Vision Model",
                [m.id for m in vision_models],
                index=0,
                key="selected_vision_model_select"
            )
            st.session_state.selected_vision_model = selected_vision
        
        # Show model details
        with st.expander("📋 Model Details"):
            for model in models[:10]:  # Show first 10
                capabilities_str = ", ".join([c.value for c in model.capabilities])
                st.write(f"**{model.name}** - {capabilities_str}")
                if model.description:
                    st.caption(model.description)
    
    else:
        # Manual model entry as fallback
        st.write("**Manual Model Configuration:**")
        
        manual_chat_model = st.text_input(
            "Chat Model Name",
            value=st.session_state.get("selected_chat_model", ""),
            placeholder="e.g., gpt-4, claude-3-sonnet, llama-2-7b-chat",
            help="Enter the exact model name for chat"
        )
        if manual_chat_model:
            st.session_state.selected_chat_model = manual_chat_model
        
        manual_vision_model = st.text_input(
            "Vision Model Name (Optional)",
            value=st.session_state.get("selected_vision_model", ""),
            placeholder="e.g., gpt-4-vision, claude-3-sonnet, llava",
            help="Enter the exact model name for image analysis"
        )
        if manual_vision_model:
            st.session_state.selected_vision_model = manual_vision_model




def render_settings():
    """Render application settings."""
    st.subheader("⚙️ Settings")
    
    # Generation parameters
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.get("temperature", 0.7),
        step=0.1,
        help="Controls randomness in responses"
    )
    st.session_state.temperature = temperature
    
    max_tokens = st.slider(
        "Max Tokens",
        min_value=256,
        max_value=4096,
        value=st.session_state.get("max_tokens", 2048),
        step=256,
        help="Maximum length of responses"
    )
    st.session_state.max_tokens = max_tokens


def render_statistics():
    """Render session statistics."""
    st.subheader("📊 Session Stats")
    
    stats = get_session_statistics()
    
    # Duration
    duration_info = stats.get("session_duration", {})
    st.metric("Session Time", duration_info.get("duration_formatted", "0s"))
    
    # Chat stats
    chat_stats = stats.get("chat_stats", {})
    if chat_stats.get("total_messages", 0) > 0:
        st.metric("Chat Messages", chat_stats["total_messages"])
        if chat_stats.get("avg_response_time", 0) > 0:
            st.metric("Avg Response", f"{chat_stats['avg_response_time']:.1f}s")
    
    # RAG chat stats
    try:
        rag_stats = get_rag_chat_statistics()
        if rag_stats.get("total_messages", 0) > 0:
            st.metric("RAG Messages", rag_stats["total_messages"])
            if rag_stats.get("rag_messages", 0) > 0:
                st.metric("RAG Enhanced", rag_stats["rag_messages"])
    except Exception:
        pass
    
    # Document stats
    try:
        doc_stats = get_document_statistics()
        if doc_stats.get("total_documents", 0) > 0:
            st.metric("Documents", doc_stats["total_documents"])
            disk_mb = doc_stats.get("disk_usage_mb", 0)
            if disk_mb > 0:
                st.metric("Storage", f"{disk_mb:.1f} MB")
    except Exception:
        pass
    
    # Image stats
    image_stats = stats.get("image_stats", {})
    if image_stats.get("total_images", 0) > 0:
        st.metric("Images Analyzed", image_stats["total_analyses"])
        if image_stats.get("avg_analysis_time", 0) > 0:
            st.metric("Avg Analysis", f"{image_stats['avg_analysis_time']:.1f}s")


def render_sidebar_actions():
    """Render sidebar action buttons."""
    st.subheader("🔧 Actions")
    
    if st.button("🗑️ Clear Chat Data", help="Clear all chat and image data"):
        clear_all_session_data()
        # Also clear RAG chat data
        if "rag_chat_messages" in st.session_state:
            st.session_state.rag_chat_messages = []
        st.success("All chat data cleared!")
        st.rerun()
    
    if st.button("📊 Export Session", help="Export session data"):
        export_session_data()
    
    # RAG-specific actions
    st.divider()
    st.write("**RAG Actions**")
    
    if st.button("🔄 Rebuild Search Index", help="Rebuild document search index"):
        try:
            from ui_components.document_interface import initialize_rag_system
            initialize_rag_system()
            st.session_state.search_engine.build_index_from_storage()
            st.success("Search index rebuilt!")
        except Exception as e:
            st.error(f"Failed to rebuild index: {str(e)}")
    
    # Memory warning
    if should_show_memory_warning():
        st.warning("⚠️ High memory usage detected")
        if st.button("🧹 Clean Up", help="Remove old data"):
            cleanup_old_data()
            st.success("Old data cleaned up!")
            st.rerun()


def export_session_data():
    """Export session data for download."""
    from utils.session_manager import export_session_data
    from datetime import datetime
    
    export_data = export_session_data()
    
    st.download_button(
        label="📥 Download Session Data",
        data=export_data,
        file_name=f"ollama_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )


def render_main_content():
    """Render main content area with tabs."""
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Chat", "🔍 RAG Chat", "📄 Documents", "🖼️ Image Analysis", "🔧 Models"])
    
    with tab1:
        render_chat_tab()
    
    with tab2:
        render_rag_chat_tab()
    
    with tab3:
        render_documents_tab()
    
    with tab4:
        render_image_tab()
    
    with tab5:
        render_models_tab()


def render_chat_tab():
    """Render standard chat interface tab."""
    # Check if API is configured
    api_endpoint = st.session_state.get("api_endpoint")
    if not api_endpoint:
        st.warning("Please configure your API endpoint in the sidebar.")
        return
    
    # Check if model is selected
    selected_model = st.session_state.get("selected_chat_model")
    if not selected_model:
        st.warning("Please select a chat model in the sidebar.")
        return
    
    # Show current configuration
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"💬 Standard Chat Mode • Using model: **{selected_model}**")
    with col2:
        if st.button("🔍 Switch to RAG Chat"):
            st.session_state.active_tab = "rag_chat"
            st.rerun()
    
    # Render chat interface
    render_chat_interface()


def render_rag_chat_tab():
    """Render RAG-enhanced chat interface tab."""
    # Check if API is configured
    api_endpoint = st.session_state.get("api_endpoint")
    if not api_endpoint:
        st.warning("Please configure your API endpoint in the sidebar.")
        return
    
    # Check if model is selected
    selected_model = st.session_state.get("selected_chat_model")
    if not selected_model:
        st.warning("Please select a chat model in the sidebar.")
        return
    
    # Show current configuration
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"🔍 RAG-Enhanced Chat Mode • Using model: **{selected_model}**")
    with col2:
        if st.button("💬 Switch to Standard Chat"):
            st.session_state.active_tab = "standard_chat"
            st.rerun()
    
    # Render RAG chat interface
    render_rag_chat_interface()


def render_documents_tab():
    """Render documents management tab."""
    render_document_interface()


def render_image_tab():
    """Render image analysis interface tab."""
    # Check if API is configured
    api_endpoint = st.session_state.get("api_endpoint")
    if not api_endpoint:
        st.warning("Please configure your API endpoint in the sidebar.")
        return
    
    # Check if vision model is selected
    selected_model = st.session_state.get("selected_vision_model")
    if not selected_model:
        st.warning("Please select a vision model in the sidebar.")
        return
    
    # Display current model
    st.info(f"Using vision model: **{selected_model}**")
    
    # Render image interface
    render_image_interface()


def render_models_tab():
    """Render model management tab."""
    render_model_management()


def handle_performance_monitoring():
    """Handle performance monitoring and cleanup."""
    # Automatic cleanup based on memory usage
    if should_show_memory_warning():
        # Auto cleanup if memory usage is very high
        cleanup_old_data(max_messages=50, max_images=5)


if __name__ == "__main__":
    main()