"""Model selector component for the FIPS Chat application."""

import streamlit as st
from typing import List, Optional
from ollama_client import get_ollama_client, ModelInfo
from config import get_model_info, RECOMMENDED_MODELS, is_vision_model


def render_model_selector(key_suffix: str = "") -> Optional[str]:
    """Render model selector with information and return selected model."""
    client = get_ollama_client()
    
    # Get available models
    available_models = client.get_available_models()
    
    if not available_models:
        st.error("No models available. Please install Ollama models first.")
        st.info("Install recommended models with: `ollama pull llava:7b` and `ollama pull granite3.3:8b`")
        return None
    
    # Extract model names
    model_names = [model.name for model in available_models]
    
    # Select model
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_model = st.selectbox(
            "Select Model",
            model_names,
            key=f"model_selector_{key_suffix}",
            help="Choose an Ollama model for this task"
        )
    
    with col2:
        if st.button("🔄 Refresh", key=f"refresh_models_{key_suffix}"):
            # Clear the client cache to force refresh
            if "ollama_client" in st.session_state:
                del st.session_state.ollama_client
            st.rerun()
    
    # Display model information
    if selected_model:
        display_model_info(selected_model, available_models)
    
    return selected_model


def display_model_info(model_name: str, available_models: List[ModelInfo]):
    """Display information about the selected model."""
    # Find the model in available models
    model_info = None
    for model in available_models:
        if model.name == model_name:
            model_info = model
            break
    
    if not model_info:
        return
    
    # Get recommended model info
    recommended_info = get_model_info(model_name)
    
    # Display model details
    with st.expander("Model Information", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Name:** {model_name}")
            st.write(f"**Type:** {recommended_info.type.title()}")
            if model_info.size > 0:
                size_gb = model_info.size / (1024**3)
                st.write(f"**Size:** {size_gb:.1f} GB")
        
        with col2:
            st.write(f"**Vision Support:** {'Yes' if is_vision_model(model_name) else 'No'}")
            if recommended_info.description != "Model information not available":
                st.write(f"**Description:** {recommended_info.description}")
        
        if model_info.modified_at:
            st.write(f"**Last Modified:** {model_info.modified_at}")


def render_model_management():
    """Render model management interface."""
    st.subheader("Model Management")
    
    client = get_ollama_client()
    available_models = client.get_available_models()
    
    # Available models section
    st.write("### Available Models")
    if available_models:
        for model in available_models:
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                st.write(f"**{model.name}**")
            
            with col2:
                if model.size > 0:
                    size_gb = model.size / (1024**3)
                    st.write(f"{size_gb:.1f} GB")
            
            with col3:
                model_type = get_model_info(model.name).type
                if model_type == "vision":
                    st.write("👁️ Vision")
                elif model_type == "code":
                    st.write("💻 Code")
                else:
                    st.write("💬 Chat")
    else:
        st.info("No models installed.")
    
    # Recommended models section
    st.write("### Recommended Models")
    st.write("These models are optimized for the application:")
    
    for model in RECOMMENDED_MODELS:
        col1, col2, col3, col4 = st.columns([3, 1, 1, 2])
        
        with col1:
            st.write(f"**{model.name}**")
        
        with col2:
            st.write(model.size)
        
        with col3:
            if model.type == "vision":
                st.write("👁️")
            elif model.type == "code":
                st.write("💻")
            else:
                st.write("💬")
        
        with col4:
            # Check if model is installed
            is_installed = any(m.name == model.name for m in available_models)
            if is_installed:
                st.success("✅ Installed")
            else:
                if st.button(f"Install", key=f"install_{model.name.replace(':', '_')}"):
                    install_model(model.name)
        
        st.write(f"*{model.description}*")
        st.write("")


def install_model(model_name: str):
    """Install a model using Ollama pull."""
    client = get_ollama_client()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner(f"Installing {model_name}..."):
        try:
            for status in client.pull_model(model_name):
                status_text.text(f"Status: {status}")
                # Simple progress indication
                if "downloading" in status.lower():
                    progress_bar.progress(0.5)
                elif "verifying" in status.lower():
                    progress_bar.progress(0.8)
                elif "success" in status.lower():
                    progress_bar.progress(1.0)
            
            st.success(f"Successfully installed {model_name}")
            progress_bar.empty()
            status_text.empty()
            
            # Clear client cache to refresh model list
            if "ollama_client" in st.session_state:
                del st.session_state.ollama_client
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Failed to install {model_name}: {str(e)}")
            progress_bar.empty()
            status_text.empty()