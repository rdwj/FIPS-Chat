"""Image analysis interface component for the FIPS Chat application."""

import streamlit as st
from PIL import Image
import io
from typing import Optional
import time
from datetime import datetime

from ai_client import get_ai_client
from config import get_config, is_vision_capable_model


def initialize_image_session():
    """Initialize image analysis session state."""
    if "uploaded_images" not in st.session_state:
        st.session_state.uploaded_images = []
    if "image_analyses" not in st.session_state:
        st.session_state.image_analyses = []
    if "vision_model" not in st.session_state:
        config = get_config()
        st.session_state.vision_model = config.default_vision_model


def render_image_interface():
    """Render the image analysis interface."""
    initialize_image_session()
    
    # Image upload section
    render_image_upload()
    
    # Current image display and analysis
    if st.session_state.uploaded_images:
        render_current_image()
    
    # Image history
    render_image_history()


def render_image_upload():
    """Render image upload interface."""
    st.subheader("Image Analysis")
    
    config = get_config()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=config.supported_image_formats,
        help=f"Supported formats: {', '.join(config.supported_image_formats).upper()}"
    )
    
    if uploaded_file is not None:
        # Check file size
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        
        if file_size_mb > config.max_file_size_mb:
            st.error(f"File size ({file_size_mb:.1f} MB) exceeds maximum allowed size ({config.max_file_size_mb} MB)")
            return
        
        # Process uploaded image
        process_uploaded_image(uploaded_file)


def process_uploaded_image(uploaded_file):
    """Process and store uploaded image."""
    try:
        # Read image
        image_data = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_data))
        
        # Create image record
        image_record = {
            "name": uploaded_file.name,
            "size": len(image_data),
            "format": image.format,
            "dimensions": image.size,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data": image_data,
            "image": image
        }
        
        # Add to session state
        st.session_state.uploaded_images.append(image_record)
        
        # Show success message
        st.success(f"Image '{uploaded_file.name}' uploaded successfully!")
        
        # Automatically analyze the image
        analyze_current_image()
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")


def render_current_image():
    """Render current image display and analysis interface."""
    if not st.session_state.uploaded_images:
        return
    
    current_image = st.session_state.uploaded_images[-1]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Current Image")
        
        # Display image
        st.image(current_image["image"], caption=current_image["name"], use_column_width=True)
        
        # Image metadata
        with st.expander("Image Details"):
            st.write(f"**Filename:** {current_image['name']}")
            st.write(f"**Format:** {current_image['format']}")
            st.write(f"**Dimensions:** {current_image['dimensions'][0]} x {current_image['dimensions'][1]} pixels")
            st.write(f"**File Size:** {current_image['size'] / 1024:.1f} KB")
            st.write(f"**Uploaded:** {current_image['timestamp']}")
        
        # Manual analysis controls
        if st.button("🔍 Analyze Image", key="analyze_current"):
            analyze_current_image()
        
    with col2:
        st.subheader("Analysis & Questions")
        
        # Custom prompt input
        custom_prompt = st.text_area(
            "Ask a question about the image:",
            placeholder="Describe this image in detail...",
            height=100
        )
        
        if st.button("Ask Question") and custom_prompt:
            analyze_image_with_prompt(current_image, custom_prompt)
        
        # Quick analysis buttons
        st.write("**Quick Analysis:**")
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("📝 Describe"):
                analyze_image_with_prompt(current_image, "Describe this image in detail.")
        
        with col_b:
            if st.button("🔍 What's this?"):
                analyze_image_with_prompt(current_image, "What is this image showing? Identify the main objects, people, or scenes.")
        
        col_c, col_d = st.columns(2)
        
        with col_c:
            if st.button("🎨 Analyze art"):
                analyze_image_with_prompt(current_image, "Analyze this image from an artistic perspective - composition, colors, style, and mood.")
        
        with col_d:
            if st.button("📊 Extract text"):
                analyze_image_with_prompt(current_image, "Extract and transcribe any text visible in this image.")


def analyze_current_image():
    """Analyze the current image with default prompt."""
    if not st.session_state.uploaded_images:
        return
    
    current_image = st.session_state.uploaded_images[-1]
    default_prompt = "Describe this image in detail, including what you see, the setting, colors, and any notable features."
    
    analyze_image_with_prompt(current_image, default_prompt)


def analyze_image_with_prompt(image_record, prompt: str):
    """Analyze image with custom prompt."""
    client = get_ai_client()
    
    if not client:
        st.error("Please configure your API endpoint first.")
        return
    
    # Get selected vision model
    selected_model = st.session_state.get("selected_vision_model")
    
    if not selected_model:
        st.error("Please select a vision model first.")
        return
    
    # Check if model likely supports vision
    if not is_vision_capable_model(selected_model):
        st.warning(f"Model '{selected_model}' may not support image analysis. Results may vary.")
    
    # Create analysis record
    analysis_record = {
        "image_name": image_record["name"],
        "prompt": prompt,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "model": selected_model,
        "response": "",
        "response_time": 0
    }
    
    # Display analysis in progress
    with st.spinner("Analyzing image..."):
        analysis_placeholder = st.empty()
        
        try:
            start_time = time.time()
            full_response = ""
            
            # Generate image analysis using unified client
            for chunk in client.generate_with_image(
                selected_model, 
                prompt, 
                image_record["data"], 
                stream=True,
                temperature=st.session_state.get("temperature", 0.7),
                max_tokens=st.session_state.get("max_tokens", 2048)
            ):
                full_response += chunk
                analysis_placeholder.markdown(f"**Analysis:** {full_response}▌")
            
            # Final response
            response_time = time.time() - start_time
            analysis_placeholder.markdown(f"**Analysis:** {full_response}")
            
            # Update analysis record
            analysis_record["response"] = full_response
            analysis_record["response_time"] = response_time
            
            # Add to session state
            st.session_state.image_analyses.append(analysis_record)
            
            # Show response time
            st.caption(f"*Analysis completed in {response_time:.1f}s*")
            
        except Exception as e:
            error_message = f"Error analyzing image: {str(e)}"
            analysis_placeholder.error(error_message)
            
            # Add error to history
            analysis_record["response"] = error_message
            st.session_state.image_analyses.append(analysis_record)


def render_image_history():
    """Render image upload and analysis history."""
    if not st.session_state.uploaded_images and not st.session_state.image_analyses:
        return
    
    st.subheader("Session History")
    
    # Controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🗑️ Clear Images"):
            st.session_state.uploaded_images = []
            st.session_state.image_analyses = []
            st.rerun()
    
    with col2:
        if st.button("💾 Export Results"):
            export_image_analyses()
    
    with col3:
        if st.session_state.image_analyses:
            st.caption(f"Total analyses: {len(st.session_state.image_analyses)}")
    
    # Display history
    if st.session_state.image_analyses:
        for i, analysis in enumerate(reversed(st.session_state.image_analyses)):
            with st.expander(f"Analysis {len(st.session_state.image_analyses) - i}: {analysis['image_name']} ({analysis['timestamp']})"):
                st.write(f"**Prompt:** {analysis['prompt']}")
                st.write(f"**Model:** {analysis['model']}")
                st.write(f"**Response:** {analysis['response']}")
                if analysis['response_time'] > 0:
                    st.write(f"**Response Time:** {analysis['response_time']:.1f}s")


def export_image_analyses():
    """Export image analyses to downloadable format."""
    if not st.session_state.image_analyses:
        st.warning("No analyses to export.")
        return
    
    # Create export content
    export_content = "# Image Analysis Export\n\n"
    export_content += f"Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for i, analysis in enumerate(st.session_state.image_analyses, 1):
        export_content += f"## Analysis {i}: {analysis['image_name']}\n\n"
        export_content += f"**Timestamp:** {analysis['timestamp']}\n\n"
        export_content += f"**Model:** {analysis['model']}\n\n"
        export_content += f"**Prompt:** {analysis['prompt']}\n\n"
        export_content += f"**Response:** {analysis['response']}\n\n"
        
        if analysis['response_time'] > 0:
            export_content += f"**Response Time:** {analysis['response_time']:.1f}s\n\n"
        
        export_content += "---\n\n"
    
    # Provide download
    st.download_button(
        label="📥 Download Analyses",
        data=export_content,
        file_name=f"image_analyses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )


def get_image_statistics():
    """Get image session statistics."""
    stats = {
        "total_images": len(st.session_state.uploaded_images),
        "total_analyses": len(st.session_state.image_analyses),
    }
    
    # Calculate average response time
    response_times = [analysis.get("response_time", 0) for analysis in st.session_state.image_analyses if analysis.get("response_time", 0) > 0]
    if response_times:
        stats["avg_response_time"] = sum(response_times) / len(response_times)
    
    return stats