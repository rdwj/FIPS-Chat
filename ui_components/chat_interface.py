"""Chat interface component for the FIPS Chat application."""

import streamlit as st
from typing import List, Dict
import time
from datetime import datetime

from ai_client import get_ai_client
from config import get_config


def initialize_chat_session():
    """Initialize chat session state."""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_model" not in st.session_state:
        config = get_config()
        st.session_state.chat_model = config.default_chat_model


def render_chat_interface():
    """Render the chat interface."""
    initialize_chat_session()
    
    # Chat messages container
    messages_container = st.container()
    
    # Display chat messages
    with messages_container:
        display_chat_messages()
    
    # Chat input
    render_chat_input()
    
    # Chat controls
    render_chat_controls()


def display_chat_messages():
    """Display chat message history."""
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Show timestamp
            if "timestamp" in message:
                st.caption(f"*{message['timestamp']}*")


def render_chat_input():
    """Render chat input field."""
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message
        timestamp = datetime.now().strftime("%H:%M:%S")
        user_message = {
            "role": "user",
            "content": prompt,
            "timestamp": timestamp
        }
        st.session_state.chat_messages.append(user_message)
        
        # Display user message immediately
        with st.chat_message("user"):
            st.write(prompt)
            st.caption(f"*{timestamp}*")
        
        # Generate assistant response
        generate_chat_response(prompt)


def generate_chat_response(user_input: str):
    """Generate and display assistant response."""
    client = get_ai_client()
    
    if not client:
        st.error("Please configure your API endpoint first.")
        return
    
    selected_model = st.session_state.get("selected_chat_model")
    if not selected_model:
        st.error("Please select a chat model first.")
        return
    
    # Prepare messages for API
    messages = []
    for msg in st.session_state.chat_messages:
        if msg["role"] in ["user", "assistant"]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    # Display assistant message
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Start timing
            start_time = time.time()
            
            # Generate response using unified client
            for chunk in client.chat(
                selected_model, 
                messages, 
                stream=True,
                temperature=st.session_state.get("temperature", 0.7),
                max_tokens=st.session_state.get("max_tokens", 2048)
            ):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            # Final response without cursor
            message_placeholder.markdown(full_response)
            
            # Calculate response time
            response_time = time.time() - start_time
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Show response time
            st.caption(f"*{timestamp} • Response time: {response_time:.1f}s*")
            
            # Add assistant message to history
            assistant_message = {
                "role": "assistant",
                "content": full_response,
                "timestamp": timestamp,
                "response_time": response_time
            }
            st.session_state.chat_messages.append(assistant_message)
            
        except Exception as e:
            error_message = f"Error generating response: {str(e)}"
            message_placeholder.error(error_message)
            
            # Add error to history
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": error_message,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })


def render_chat_controls():
    """Render chat control buttons."""
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    
    with col1:
        if st.button("🗑️ Clear Chat", help="Clear conversation history"):
            st.session_state.chat_messages = []
            st.rerun()
    
    with col2:
        if st.button("📋 Copy Last", help="Copy last response to clipboard"):
            copy_last_response()
    
    with col3:
        if st.button("💾 Export", help="Export conversation"):
            export_conversation()
    
    with col4:
        # Display conversation stats
        if st.session_state.chat_messages:
            total_messages = len(st.session_state.chat_messages)
            user_messages = sum(1 for msg in st.session_state.chat_messages if msg["role"] == "user")
            st.caption(f"Messages: {total_messages} (You: {user_messages})")


def copy_last_response():
    """Copy the last assistant response to clipboard."""
    assistant_messages = [msg for msg in st.session_state.chat_messages if msg["role"] == "assistant"]
    
    if assistant_messages:
        last_response = assistant_messages[-1]["content"]
        # Note: Actual clipboard functionality would require additional JavaScript
        # For now, display the content in a text area that users can copy
        st.text_area(
            "Last Response (select all and copy):",
            value=last_response,
            height=200,
            key="copy_response"
        )
    else:
        st.warning("No assistant responses to copy.")


def export_conversation():
    """Export conversation to downloadable format."""
    if not st.session_state.chat_messages:
        st.warning("No conversation to export.")
        return
    
    # Create markdown export
    export_content = "# Chat Conversation Export\n\n"
    export_content += f"Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for message in st.session_state.chat_messages:
        role = "You" if message["role"] == "user" else "Assistant"
        timestamp = message.get("timestamp", "")
        export_content += f"## {role}"
        if timestamp:
            export_content += f" ({timestamp})"
        export_content += "\n\n"
        export_content += f"{message['content']}\n\n"
        
        # Add response time if available
        if "response_time" in message:
            export_content += f"*Response time: {message['response_time']:.1f}s*\n\n"
        
        export_content += "---\n\n"
    
    # Provide download
    st.download_button(
        label="📥 Download Conversation",
        data=export_content,
        file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )




def get_chat_statistics():
    """Get chat session statistics."""
    if not st.session_state.chat_messages:
        return {}
    
    stats = {
        "total_messages": len(st.session_state.chat_messages),
        "user_messages": sum(1 for msg in st.session_state.chat_messages if msg["role"] == "user"),
        "assistant_messages": sum(1 for msg in st.session_state.chat_messages if msg["role"] == "assistant"),
    }
    
    # Calculate average response time
    response_times = [msg.get("response_time", 0) for msg in st.session_state.chat_messages if "response_time" in msg]
    if response_times:
        stats["avg_response_time"] = sum(response_times) / len(response_times)
    
    return stats