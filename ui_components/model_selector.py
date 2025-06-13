"""Model selector component for the FIPS Chat application."""

import streamlit as st
from typing import List, Optional
from ai_client import get_ai_client, ModelInfo, ModelCapability
from config import is_vision_capable_model, is_code_capable_model


def render_model_selector(key_suffix: str = "") -> Optional[str]:
    """Render model selector with information and return selected model."""
    client = get_ai_client()
    
    if not client:
        st.error("Please configure your API endpoint first.")
        return None
    
    # Get available models
    try:
        available_models = client.discover_models()
    except Exception as e:
        st.error(f"Failed to discover models: {str(e)}")
        return None
    
    if not available_models:
        st.warning("No models discovered. Please check your API endpoint.")
        return None
    
    # Extract model names
    model_names = [model.id for model in available_models]
    
    # Select model
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_model = st.selectbox(
            "Select Model",
            model_names,
            key=f"model_selector_{key_suffix}",
            help="Choose a model for this task"
        )
    
    with col2:
        if st.button("🔄 Refresh", key=f"refresh_models_{key_suffix}"):
            # Force refresh of model cache
            client.discover_models(force_refresh=True)
            st.rerun()
    
    # Display model information
    if selected_model:
        display_model_info(selected_model, available_models)
    
    return selected_model


def display_model_info(model_id: str, available_models: List[ModelInfo]):
    """Display information about the selected model."""
    # Find the model in available models
    model_info = None
    for model in available_models:
        if model.id == model_id:
            model_info = model
            break
    
    if not model_info:
        return
    
    # Display model details
    with st.expander("Model Information", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**ID:** {model_info.id}")
            st.write(f"**Name:** {model_info.name}")
            if model_info.capabilities:
                capabilities_str = ", ".join([c.value.title() for c in model_info.capabilities])
                st.write(f"**Capabilities:** {capabilities_str}")
        
        with col2:
            if model_info.context_length:
                st.write(f"**Context Length:** {model_info.context_length:,}")
            if model_info.size:
                st.write(f"**Size:** {model_info.size}")
            
            # Inferred capabilities
            vision_support = (ModelCapability.VISION in model_info.capabilities or 
                            is_vision_capable_model(model_info.id))
            code_support = (ModelCapability.CODE in model_info.capabilities or 
                          is_code_capable_model(model_info.id))
            
            st.write(f"**Vision Support:** {'Yes' if vision_support else 'No'}")
            st.write(f"**Code Optimized:** {'Yes' if code_support else 'No'}")
        
        if model_info.description:
            st.write(f"**Description:** {model_info.description}")


def render_model_management():
    """Render model management interface."""
    st.subheader("Model Discovery & Management")
    
    client = get_ai_client()
    
    if not client:
        st.warning("Please configure your API endpoint to discover models.")
        return
    
    # Model discovery controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("### Discovered Models")
    
    with col2:
        if st.button("🔄 Refresh All Models"):
            with st.spinner("Discovering models..."):
                try:
                    models = client.discover_models(force_refresh=True)
                    if models:
                        st.success(f"Found {len(models)} models")
                    else:
                        st.warning("No models found")
                    st.rerun()
                except Exception as e:
                    st.error(f"Discovery failed: {str(e)}")
    
    # Get available models
    try:
        available_models = client.discover_models()
    except Exception as e:
        st.error(f"Failed to get models: {str(e)}")
        return
    
    if available_models:
        # Group models by capability
        chat_models = []
        vision_models = []
        code_models = []
        other_models = []
        
        for model in available_models:
            if ModelCapability.VISION in model.capabilities:
                vision_models.append(model)
            elif ModelCapability.CODE in model.capabilities:
                code_models.append(model)
            elif ModelCapability.CHAT in model.capabilities:
                chat_models.append(model)
            else:
                other_models.append(model)
        
        # Display models by category
        if chat_models:
            st.write("#### 💬 Chat Models")
            for model in chat_models[:10]:  # Show first 10
                display_model_card(model)
        
        if vision_models:
            st.write("#### 👁️ Vision Models")
            for model in vision_models[:10]:
                display_model_card(model)
        
        if code_models:
            st.write("#### 💻 Code Models")
            for model in code_models[:10]:
                display_model_card(model)
        
        if other_models:
            st.write("#### 🔧 Other Models")
            for model in other_models[:5]:
                display_model_card(model)
        
        if len(available_models) > 35:
            st.info(f"Showing first 35 models. Total discovered: {len(available_models)}")
    
    else:
        st.info("No models discovered. Please check your API endpoint configuration.")
        
        # Show API templates as suggestions
        from config import get_all_api_templates
        templates = get_all_api_templates()
        
        st.write("### Suggested API Endpoints:")
        for name, template in templates.items():
            with st.expander(f"{template['name']} - {name}"):
                st.write(f"**Endpoint:** `{template['endpoint']}`")
                st.write(f"**Requires API Key:** {'Yes' if template['requires_key'] else 'No'}")
                if template.get('example_models'):
                    st.write(f"**Example Models:** {', '.join(template['example_models'])}")


def display_model_card(model: ModelInfo):
    """Display a compact model card."""
    col1, col2, col3 = st.columns([3, 2, 1])
    
    with col1:
        st.write(f"**{model.name}**")
        st.caption(model.id)
    
    with col2:
        if model.size:
            st.write(model.size)
        elif model.context_length:
            st.write(f"{model.context_length:,} ctx")
    
    with col3:
        # Show capability icons
        icons = []
        if ModelCapability.CHAT in model.capabilities:
            icons.append("💬")
        if ModelCapability.VISION in model.capabilities:
            icons.append("👁️")
        if ModelCapability.CODE in model.capabilities:
            icons.append("💻")
        if ModelCapability.EMBEDDING in model.capabilities:
            icons.append("🔢")
        
        if icons:
            st.write(" ".join(icons))
    
    if model.description:
        st.caption(model.description)