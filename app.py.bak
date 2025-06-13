"""FIPS Chat - Multi-provider AI chat and image analysis platform."""

import streamlit as st
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config, RECOMMENDED_MODELS
from ui_components.model_selector import render_model_selector, render_model_management
from ui_components.chat_interface import render_chat_interface, get_chat_statistics
from ui_components.image_interface import render_image_interface, get_image_statistics
from utils.session_manager import (
    initialize_session_state, 
    get_session_statistics, 
    clear_all_session_data,
    should_show_memory_warning,
    cleanup_old_data
)
from ollama_client import get_ollama_client
from api_client import get_api_client, test_api_connection


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
    check_ollama_connection()


def check_ollama_connection():
    """Check and display connection status based on selected provider."""
    provider = st.session_state.get("api_provider", "ollama")
    
    if provider == "ollama":
        try:
            client = get_ollama_client()
            models = client.get_available_models()
            
            if models:
                st.success(f"✅ Connected to Ollama • {len(models)} models available")
            else:
                st.warning("⚠️ Connected to Ollama but no models found")
                
        except Exception as e:
            st.error(f"❌ Cannot connect to Ollama: {str(e)}")
            st.info("Make sure Ollama is running: `ollama serve`")
    else:
        # Check external API connection
        api_endpoint = st.session_state.get("api_endpoint")
        api_key = st.session_state.get("api_key")
        
        if api_endpoint and api_key:
            try:
                is_connected, message = test_api_connection(provider, api_endpoint, api_key)
                if is_connected:
                    st.success(f"✅ Connected to {provider.upper()} API")
                else:
                    st.error(f"❌ {message}")
            except Exception as e:
                st.error(f"❌ API connection error: {str(e)}")
        else:
            if provider == "agentic_pipeline":
                st.warning("⚠️ Please configure your agentic pipeline endpoint")
            else:
                st.warning("⚠️ Please configure API endpoint and key")


def render_sidebar():
    """Render sidebar with controls and information."""
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # API provider selection
        render_api_provider_selector()
        
        st.divider()
        
        # Model selectors (conditional based on provider)
        if st.session_state.get("api_provider") == "ollama":
            st.subheader("Chat Model")
            selected_chat_model = render_model_selector("chat")
            if selected_chat_model:
                st.session_state.selected_chat_model = selected_chat_model
            
            st.subheader("Vision Model")
            selected_vision_model = render_model_selector("vision")
            if selected_vision_model:
                st.session_state.selected_vision_model = selected_vision_model
        else:
            render_external_api_models()
        
        st.divider()
        
        # Settings
        render_settings()
        
        st.divider()
        
        # Statistics
        render_statistics()
        
        st.divider()
        
        # Actions
        render_sidebar_actions()


def render_api_provider_selector():
    """Render API provider selection."""
    st.subheader("🌐 API Provider")
    
    provider = st.selectbox(
        "Select Provider",
        ["ollama", "agentic_pipeline", "openai_compatible", "custom"],
        index=0 if st.session_state.get("api_provider", "ollama") == "ollama" else 
              1 if st.session_state.get("api_provider") == "agentic_pipeline" else
              2 if st.session_state.get("api_provider") == "openai_compatible" else 3,
        help="Choose your AI API provider"
    )
    st.session_state.api_provider = provider
    
    if provider != "ollama":
        render_external_api_config()


def render_external_api_config():
    """Render external API configuration."""
    provider = st.session_state.get("api_provider")
    
    # API Endpoint
    if provider == "agentic_pipeline":
        st.markdown("**🤖 Agentic Pipeline Configuration:**")
        st.caption("Connect to your external agentic application backend")
        placeholder = "http://your-agent-service:8080/api/chat"
    elif provider == "openai_compatible":
        st.markdown("**For vLLM, Ollama API, or other OpenAI-compatible APIs:**")
        placeholder = "http://your-vllm-service:8000/v1"
    else:
        st.markdown("**Custom API Configuration:**")
        placeholder = "https://your-api-endpoint.com"
    
    api_endpoint = st.text_input(
        "API Endpoint",
        value=st.session_state.get("api_endpoint", ""),
        placeholder=placeholder,
        help="Full URL to your API endpoint"
    )
    st.session_state.api_endpoint = api_endpoint
    
    # API Key (conditional based on provider)
    if provider == "agentic_pipeline":
        api_key = st.text_input(
            "API Key (Optional)",
            value=st.session_state.get("api_key", ""),
            type="password",
            placeholder="Bearer token or auth key (if required)",
            help="Authentication key for your agentic pipeline"
        )
    else:
        api_key = st.text_input(
            "API Key",
            value=st.session_state.get("api_key", ""),
            type="password",
            placeholder="Your API key (leave empty if not required)",
            help="API authentication key"
        )
    st.session_state.api_key = api_key
    
    # API Type for custom providers
    if provider == "custom":
        api_type = st.selectbox(
            "API Type",
            ["openai_compatible", "anthropic", "custom"],
            help="Select the API format your endpoint uses"
        )
        st.session_state.api_type = api_type


def render_external_api_models():
    """Render model selection for external APIs."""
    provider = st.session_state.get("api_provider")
    
    if provider == "agentic_pipeline":
        st.subheader("Pipeline Configuration")
        
        # Pipeline ID/Name (optional)
        pipeline_id = st.text_input(
            "Pipeline ID (Optional)",
            value=st.session_state.get("external_model_name", ""),
            placeholder="e.g., main, production, v2.0",
            help="Specific pipeline version or configuration to use"
        )
        st.session_state.external_model_name = pipeline_id
        
        # Session ID for conversation continuity
        session_id = st.text_input(
            "Session ID (Optional)",
            value=st.session_state.get("agent_session_id", ""),
            placeholder="Leave empty for auto-generated session",
            help="Session ID for conversation continuity across requests"
        )
        st.session_state.agent_session_id = session_id
        
    else:
        st.subheader("Model Configuration")
        
        # Model name input
        model_name = st.text_input(
            "Model Name",
            value=st.session_state.get("external_model_name", ""),
            placeholder="e.g., gpt-4, llama-2-7b-chat, claude-3-sonnet",
            help="The exact model name as required by your API"
        )
        st.session_state.external_model_name = model_name
    
    # Test connection button
    if st.button("🔍 Test Connection"):
        test_external_api_connection()


def test_external_api_connection():
    """Test external API connection and show results."""
    provider = st.session_state.get("api_provider")
    endpoint = st.session_state.get("api_endpoint")
    api_key = st.session_state.get("api_key")
    model_name = st.session_state.get("external_model_name")
    
    if not endpoint:
        st.error("Please enter an API endpoint")
        return
    
    if not model_name:
        st.error("Please enter a model name")
        return
    
    try:
        with st.spinner("Testing connection..."):
            is_connected, message = test_api_connection(provider, endpoint, api_key, model_name)
            
        if is_connected:
            st.success(f"✅ Connection successful: {message}")
        else:
            st.error(f"❌ Connection failed: {message}")
            
    except Exception as e:
        st.error(f"❌ Error testing connection: {str(e)}")


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
    
    # Image stats
    image_stats = stats.get("image_stats", {})
    if image_stats.get("total_images", 0) > 0:
        st.metric("Images Analyzed", image_stats["total_analyses"])
        if image_stats.get("avg_analysis_time", 0) > 0:
            st.metric("Avg Analysis", f"{image_stats['avg_analysis_time']:.1f}s")


def render_sidebar_actions():
    """Render sidebar action buttons."""
    st.subheader("🔧 Actions")
    
    if st.button("🗑️ Clear All Data", help="Clear all chat and image data"):
        clear_all_session_data()
        st.success("All data cleared!")
        st.rerun()
    
    if st.button("📊 Export Session", help="Export session data"):
        export_session_data()
    
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
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "🖼️ Image Analysis", "🔧 Models"])
    
    with tab1:
        render_chat_tab()
    
    with tab2:
        render_image_tab()
    
    with tab3:
        render_models_tab()


def render_chat_tab():
    """Render chat interface tab."""
    provider = st.session_state.get("api_provider", "ollama")
    
    # Check if model/endpoint is configured
    if provider == "ollama":
        selected_model = st.session_state.get("selected_chat_model") or st.session_state.get("chat_model")
        if not selected_model:
            st.warning("Please select a chat model from the sidebar.")
            return
        st.info(f"Using Ollama model: **{selected_model}**")
    else:
        api_endpoint = st.session_state.get("api_endpoint")
        if not api_endpoint:
            if provider == "agentic_pipeline":
                st.warning("Please configure your agentic pipeline endpoint in the sidebar.")
            else:
                st.warning("Please configure API endpoint in the sidebar.")
            return
        
        model_name = st.session_state.get("external_model_name")
        if not model_name and provider != "agentic_pipeline":
            st.warning("Please configure model name in the sidebar.")
            return
        if provider == "agentic_pipeline":
            pipeline_info = f"Pipeline: **{model_name}**" if model_name else "**Default Pipeline**"
            st.info(f"Using Agentic Pipeline - {pipeline_info}")
        else:
            st.info(f"Using {provider.upper()} model: **{model_name}**")
    
    # Render chat interface
    render_chat_interface()


def render_image_tab():
    """Render image analysis interface tab."""
    # Check if vision model is selected
    selected_model = st.session_state.get("selected_vision_model") or st.session_state.get("vision_model")
    
    if not selected_model:
        st.warning("Please select a vision model from the sidebar.")
        return
    
    # Display current model
    st.info(f"Using model: **{selected_model}**")
    
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