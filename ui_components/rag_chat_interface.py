"""RAG-enhanced chat interface component."""

import streamlit as st
import logging
from typing import List, Dict, Optional, Any, Generator
import time
from datetime import datetime

from ai_client import get_ai_client
from rag.rag_pipeline import RAGPipeline, RAGResponse, SourceCitation
from rag.file_storage import FileStorage
from rag.search_engine import TFIDFSearchEngine
from ui_components.document_interface import initialize_rag_system, check_rag_system_available

logger = logging.getLogger(__name__)


def initialize_rag_chat():
    """Initialize RAG chat system."""
    if "rag_chat_messages" not in st.session_state:
        st.session_state.rag_chat_messages = []
    
    if "rag_enabled" not in st.session_state:
        st.session_state.rag_enabled = True
    
    if "rag_pipeline" not in st.session_state and check_rag_system_available():
        try:
            initialize_rag_system()
            ai_client = get_ai_client()
            
            if ai_client:
                st.session_state.rag_pipeline = RAGPipeline(
                    storage=st.session_state.rag_storage,
                    search_engine=st.session_state.search_engine,
                    ai_client=ai_client,
                    max_context_tokens=st.session_state.get("rag_context_length", 3000)
                )
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            st.session_state.rag_pipeline = None


def render_rag_chat_interface():
    """Render the RAG-enhanced chat interface."""
    initialize_rag_chat()
    
    # RAG status and controls
    render_rag_controls()
    
    # Chat messages container
    messages_container = st.container()
    
    # Display chat messages
    with messages_container:
        display_rag_chat_messages()
    
    # Chat input
    render_rag_chat_input()
    
    # Chat controls
    render_rag_chat_controls()


def render_rag_controls():
    """Render RAG mode controls and status."""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # RAG mode toggle
        rag_enabled = st.checkbox(
            "🔍 RAG Enhancement",
            value=st.session_state.get("rag_enabled", True),
            help="Use uploaded documents to enhance responses with relevant context"
        )
        st.session_state.rag_enabled = rag_enabled
        
        # Show RAG status
        if rag_enabled:
            if check_rag_system_available():
                doc_count = 0
                try:
                    docs = st.session_state.rag_storage.list_documents()
                    doc_count = len(docs)
                except Exception:
                    pass
                
                if doc_count > 0:
                    st.success(f"✅ RAG Active • {doc_count} documents available")
                else:
                    st.warning("⚠️ RAG enabled but no documents uploaded")
            else:
                st.error("❌ RAG system unavailable")
        else:
            st.info("ℹ️ RAG disabled - using standard chat mode")
    
    with col2:
        # RAG settings button
        if st.button("⚙️ RAG Settings"):
            st.session_state.show_rag_settings = not st.session_state.get("show_rag_settings", False)
    
    with col3:
        # RAG statistics button
        if st.button("📊 RAG Stats"):
            st.session_state.show_rag_stats = not st.session_state.get("show_rag_stats", False)
    
    # RAG settings panel
    if st.session_state.get("show_rag_settings", False):
        render_rag_settings_panel()
    
    # RAG statistics panel
    if st.session_state.get("show_rag_stats", False):
        render_rag_statistics_panel()


def render_rag_settings_panel():
    """Render RAG configuration panel."""
    with st.expander("🔍 RAG Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Search sensitivity
            relevance_threshold = st.slider(
                "Search Sensitivity",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get("rag_threshold", 0.1),
                step=0.05,
                help="Lower values include more sources (less strict)"
            )
            st.session_state.rag_threshold = relevance_threshold
            
            # Number of sources
            max_sources = st.selectbox(
                "Maximum Sources",
                options=[3, 5, 7, 10],
                index=1,  # Default to 5
                help="Maximum number of source documents to include"
            )
            st.session_state.rag_max_sources = max_sources
        
        with col2:
            # Context length
            context_length = st.selectbox(
                "Context Length",
                options=[2000, 3000, 4000, 5000],
                index=1,  # Default to 3000
                help="Maximum context tokens to include in prompts"
            )
            st.session_state.rag_context_length = context_length
            
            # RAG mode strictness
            rag_mode = st.selectbox(
                "RAG Mode",
                options=["Adaptive", "Strict", "Fallback"],
                index=0,
                help="How to handle cases with limited relevant context"
            )
            st.session_state.rag_mode = rag_mode
        
        # Update RAG pipeline if settings changed
        if st.button("Apply Settings"):
            if st.session_state.get("rag_pipeline"):
                st.session_state.rag_pipeline.context_manager.max_tokens = context_length
                st.session_state.rag_pipeline.quality_controller.min_relevance = relevance_threshold
            st.success("Settings applied!")


def render_rag_statistics_panel():
    """Render RAG statistics panel."""
    with st.expander("📊 RAG Statistics", expanded=True):
        try:
            if not st.session_state.get("rag_pipeline"):
                st.warning("RAG pipeline not initialized")
                return
            
            # Get pipeline stats
            stats = st.session_state.rag_pipeline.get_pipeline_stats()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Queries", stats.get('total_queries', 0))
                st.metric("Full RAG Responses", stats.get('full_rag_queries', 0))
            
            with col2:
                st.metric("Partial RAG Responses", stats.get('partial_rag_queries', 0))
                st.metric("Fallback Responses", stats.get('fallback_queries', 0))
            
            with col3:
                avg_time = stats.get('avg_processing_time', 0)
                st.metric("Avg Processing Time", f"{avg_time:.2f}s")
                full_rate = stats.get('full_rag_rate', 0)
                st.metric("Full RAG Rate", f"{full_rate:.1%}")
            
            # Show RAG effectiveness
            if stats.get('total_queries', 0) > 0:
                st.subheader("RAG Effectiveness")
                
                # Create a simple chart
                import pandas as pd
                mode_data = {
                    'Mode': ['Full RAG', 'Partial RAG', 'Fallback'],
                    'Count': [
                        stats.get('full_rag_queries', 0),
                        stats.get('partial_rag_queries', 0),
                        stats.get('fallback_queries', 0)
                    ]
                }
                df = pd.DataFrame(mode_data)
                st.bar_chart(df.set_index('Mode'))
        
        except Exception as e:
            st.error(f"Error loading RAG statistics: {str(e)}")


def display_rag_chat_messages():
    """Display RAG chat message history with source citations."""
    for message in st.session_state.rag_chat_messages:
        with st.chat_message(message["role"]):
            # Message content
            st.write(message["content"])
            
            # Message metadata
            col1, col2 = st.columns([3, 1])
            with col1:
                if "timestamp" in message:
                    st.caption(f"*{message['timestamp']}*")
            with col2:
                if message.get('rag_mode'):
                    mode_icon = {
                        'full': '🔍',
                        'partial': '🔎',
                        'fallback': '💬'
                    }.get(message['rag_mode'], '💬')
                    st.caption(f"{mode_icon} {message['rag_mode'].title()}")
            
            # Source citations for RAG responses
            if message["role"] == "assistant" and message.get("sources"):
                render_source_citations(message["sources"])
            
            # Processing time if available
            if message.get("processing_time"):
                st.caption(f"Processing time: {message['processing_time']:.1f}s")


def render_source_citations(sources: List[Dict[str, Any]]):
    """Display source citations for RAG responses."""
    if not sources:
        return
    
    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
        for i, source in enumerate(sources):
            st.write(f"**{i+1}. {source['document']}** (Page {source['page_number']})")
            
            # Section if available
            if source.get('section_title') and source['section_title'] != 'Content':
                st.caption(f"Section: {source['section_title']}")
            
            # Source excerpt
            excerpt = source.get('excerpt', '')
            if excerpt:
                # Truncate very long excerpts
                if len(excerpt) > 300:
                    excerpt = excerpt[:300] + "..."
                st.markdown(f"> {excerpt}")
            
            # Relevance score with visual indicator
            relevance = source.get('relevance_score', 0.0)
            st.progress(relevance, text=f"Relevance: {relevance:.2f}")
            
            # View full context button
            if st.button(f"👁️ View Context", key=f"view_context_{i}"):
                show_source_context(source)
            
            if i < len(sources) - 1:
                st.divider()


def show_source_context(source: Dict[str, Any]):
    """Show full context for a source."""
    try:
        doc_id = source.get('doc_id')
        chunk_id = source.get('chunk_id')
        
        if not doc_id:
            st.error("Document ID not available")
            return
        
        # Load document
        doc_result = st.session_state.rag_storage.load_document(doc_id)
        if not doc_result:
            st.error("Could not load document")
            return
        
        # Find the specific chunk
        target_chunk = None
        for chunk in doc_result.chunks:
            if chunk.metadata.chunk_id == chunk_id:
                target_chunk = chunk
                break
        
        if not target_chunk:
            st.error("Could not find source chunk")
            return
        
        # Display in modal-like expander
        with st.expander(f"📄 Full Context: {source['document']}", expanded=True):
            st.subheader(f"Page {source['page_number']} - {source.get('section_title', 'Content')}")
            
            # Full chunk text
            st.text_area(
                "Source Content",
                value=target_chunk.text,
                height=300,
                disabled=True,
                key=f"full_context_{chunk_id}"
            )
            
            # Chunk metadata
            st.caption(f"Chunk ID: {chunk_id}")
            st.caption(f"Word Count: {target_chunk.metadata.word_count}")
            st.caption(f"Position: {target_chunk.metadata.char_start}-{target_chunk.metadata.char_end}")
    
    except Exception as e:
        st.error(f"Error loading source context: {str(e)}")


def render_rag_chat_input():
    """Render RAG chat input field."""
    # Check if we can use RAG
    can_use_rag = (
        st.session_state.get("rag_enabled", False) and 
        check_rag_system_available() and 
        st.session_state.get("rag_pipeline") is not None
    )
    
    # Chat input with dynamic placeholder
    if can_use_rag:
        placeholder = "Ask me anything about your documents..."
    else:
        placeholder = "Ask me anything..."
    
    if prompt := st.chat_input(placeholder, key="rag_chat_input"):
        # Add user message
        timestamp = datetime.now().strftime("%H:%M:%S")
        user_message = {
            "role": "user",
            "content": prompt,
            "timestamp": timestamp
        }
        st.session_state.rag_chat_messages.append(user_message)
        
        # Display user message immediately
        with st.chat_message("user"):
            st.write(prompt)
            st.caption(f"*{timestamp}*")
        
        # Generate response (RAG or standard)
        if can_use_rag:
            generate_rag_response(prompt)
        else:
            generate_standard_response(prompt)


def generate_rag_response(user_input: str):
    """Generate RAG-enhanced response."""
    client = get_ai_client()
    
    if not client:
        st.error("Please configure your API endpoint first.")
        return
    
    selected_model = st.session_state.get("selected_chat_model")
    if not selected_model:
        st.error("Please select a chat model first.")
        return
    
    # Display assistant message
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # Start timing
            start_time = time.time()
            
            # Show processing status
            message_placeholder.info("🔍 Searching documents and generating response...")
            
            # Process query through RAG pipeline
            rag_response = st.session_state.rag_pipeline.process_rag_query(
                query=user_input,
                model=selected_model,
                max_sources=st.session_state.get("rag_max_sources", 5),
                relevance_threshold=st.session_state.get("rag_threshold", 0.1),
                stream=False,  # For now, disable streaming for RAG
                temperature=st.session_state.get("temperature", 0.7),
                max_tokens=st.session_state.get("max_tokens", 2048)
            )
            
            # Display the response
            message_placeholder.markdown(rag_response.response)
            
            # Calculate total time
            total_time = time.time() - start_time
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Show metadata
            mode_display = {
                'full': '🔍 Full RAG',
                'partial': '🔎 Partial RAG', 
                'fallback': '💬 Fallback'
            }.get(rag_response.rag_mode, '💬 Standard')
            
            st.caption(f"*{timestamp} • {mode_display} • {total_time:.1f}s*")
            
            # Add to message history
            assistant_message = {
                "role": "assistant",
                "content": rag_response.response,
                "timestamp": timestamp,
                "processing_time": total_time,
                "rag_mode": rag_response.rag_mode,
                "sources": [source.to_dict() for source in rag_response.sources],
                "context_used": rag_response.context_used,
                "relevance_scores": rag_response.relevance_scores
            }
            st.session_state.rag_chat_messages.append(assistant_message)
            
        except Exception as e:
            error_message = f"Error generating RAG response: {str(e)}"
            message_placeholder.error(error_message)
            logger.error(f"RAG response error: {e}")
            
            # Fallback to standard response
            try:
                message_placeholder.info("Falling back to standard response...")
                generate_standard_response(user_input, message_placeholder)
            except Exception as fallback_error:
                message_placeholder.error(f"Fallback also failed: {str(fallback_error)}")


def generate_standard_response(user_input: str, message_placeholder=None):
    """Generate standard (non-RAG) response."""
    client = get_ai_client()
    
    if not client:
        if message_placeholder:
            message_placeholder.error("Please configure your API endpoint first.")
        else:
            st.error("Please configure your API endpoint first.")
        return
    
    selected_model = st.session_state.get("selected_chat_model")
    if not selected_model:
        if message_placeholder:
            message_placeholder.error("Please select a chat model first.")
        else:
            st.error("Please select a chat model first.")
        return
    
    # Prepare messages for API (convert RAG messages to standard format)
    messages = []
    for msg in st.session_state.rag_chat_messages:
        if msg["role"] in ["user", "assistant"]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    # Create message placeholder if not provided
    if not message_placeholder:
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
        st.caption(f"*{timestamp} • 💬 Standard • Response time: {response_time:.1f}s*")
        
        # Add assistant message to history
        assistant_message = {
            "role": "assistant",
            "content": full_response,
            "timestamp": timestamp,
            "processing_time": response_time,
            "rag_mode": "standard"
        }
        st.session_state.rag_chat_messages.append(assistant_message)
        
    except Exception as e:
        error_message = f"Error generating response: {str(e)}"
        message_placeholder.error(error_message)
        
        # Add error to history
        st.session_state.rag_chat_messages.append({
            "role": "assistant",
            "content": error_message,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "rag_mode": "error"
        })


def render_rag_chat_controls():
    """Render RAG chat control buttons."""
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    
    with col1:
        if st.button("🗑️ Clear Chat", help="Clear conversation history", key="rag_clear_chat"):
            st.session_state.rag_chat_messages = []
            st.rerun()
    
    with col2:
        if st.button("📋 Copy Last", help="Copy last response to clipboard", key="rag_copy_last"):
            copy_last_rag_response()
    
    with col3:
        if st.button("💾 Export", help="Export conversation with sources", key="rag_export"):
            export_rag_conversation()
    
    with col4:
        # Display conversation stats
        if st.session_state.rag_chat_messages:
            total_messages = len(st.session_state.rag_chat_messages)
            user_messages = sum(1 for msg in st.session_state.rag_chat_messages if msg["role"] == "user")
            rag_messages = sum(1 for msg in st.session_state.rag_chat_messages 
                             if msg.get("rag_mode") in ["full", "partial"])
            st.caption(f"Messages: {total_messages} (You: {user_messages}, RAG: {rag_messages})")


def copy_last_rag_response():
    """Copy the last assistant response with source information."""
    assistant_messages = [msg for msg in st.session_state.rag_chat_messages if msg["role"] == "assistant"]
    
    if assistant_messages:
        last_response = assistant_messages[-1]
        
        # Build response text with sources
        response_text = last_response["content"]
        
        if last_response.get("sources"):
            response_text += "\n\n**Sources:**\n"
            for i, source in enumerate(last_response["sources"]):
                response_text += f"{i+1}. {source['document']} (Page {source['page_number']})\n"
                if source.get('excerpt'):
                    response_text += f"   \"{source['excerpt'][:100]}...\"\n"
        
        # Display in text area for copying
        st.text_area(
            "Last Response with Sources (select all and copy):",
            value=response_text,
            height=300,
            key="copy_rag_response"
        )
    else:
        st.warning("No assistant responses to copy.")


def export_rag_conversation():
    """Export RAG conversation to downloadable format with source information."""
    if not st.session_state.rag_chat_messages:
        st.warning("No conversation to export.")
        return
    
    # Create detailed markdown export
    export_content = "# RAG-Enhanced Chat Conversation Export\n\n"
    export_content += f"Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    export_content += f"RAG Mode: {'Enabled' if st.session_state.get('rag_enabled') else 'Disabled'}\n\n"
    
    for message in st.session_state.rag_chat_messages:
        role = "You" if message["role"] == "user" else "Assistant"
        timestamp = message.get("timestamp", "")
        
        export_content += f"## {role}"
        if timestamp:
            export_content += f" ({timestamp})"
        
        # Add RAG mode info for assistant messages
        if message["role"] == "assistant" and message.get("rag_mode"):
            mode_display = {
                'full': 'Full RAG',
                'partial': 'Partial RAG',
                'fallback': 'Fallback',
                'standard': 'Standard'
            }.get(message["rag_mode"], message["rag_mode"])
            export_content += f" - {mode_display}"
        
        export_content += "\n\n"
        export_content += f"{message['content']}\n\n"
        
        # Add processing time if available
        if message.get("processing_time"):
            export_content += f"*Processing time: {message['processing_time']:.1f}s*\n\n"
        
        # Add sources for RAG responses
        if message.get("sources"):
            export_content += "**Sources:**\n\n"
            for i, source in enumerate(message["sources"]):
                export_content += f"{i+1}. **{source['document']}** (Page {source['page_number']})\n"
                if source.get('section_title') and source['section_title'] != 'Content':
                    export_content += f"   Section: {source['section_title']}\n"
                if source.get('excerpt'):
                    export_content += f"   > {source['excerpt']}\n"
                export_content += f"   Relevance: {source.get('relevance_score', 0):.2f}\n\n"
        
        export_content += "---\n\n"
    
    # Add RAG statistics if available
    if st.session_state.get("rag_pipeline"):
        try:
            stats = st.session_state.rag_pipeline.get_pipeline_stats()
            export_content += "## RAG Statistics\n\n"
            export_content += f"- Total Queries: {stats.get('total_queries', 0)}\n"
            export_content += f"- Full RAG Responses: {stats.get('full_rag_queries', 0)}\n"
            export_content += f"- Partial RAG Responses: {stats.get('partial_rag_queries', 0)}\n"
            export_content += f"- Fallback Responses: {stats.get('fallback_queries', 0)}\n"
            export_content += f"- Average Processing Time: {stats.get('avg_processing_time', 0):.2f}s\n\n"
        except Exception:
            pass
    
    # Provide download
    st.download_button(
        label="📥 Download RAG Conversation",
        data=export_content,
        file_name=f"rag_chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )


def get_rag_chat_statistics():
    """Get RAG chat session statistics."""
    if not st.session_state.rag_chat_messages:
        return {}
    
    stats: Dict[str, Any] = {
        "total_messages": len(st.session_state.rag_chat_messages),
        "user_messages": sum(1 for msg in st.session_state.rag_chat_messages if msg["role"] == "user"),
        "assistant_messages": sum(1 for msg in st.session_state.rag_chat_messages if msg["role"] == "assistant"),
    }
    
    # RAG-specific stats
    rag_messages = [msg for msg in st.session_state.rag_chat_messages if msg.get("rag_mode")]
    if rag_messages:
        stats["rag_messages"] = len(rag_messages)
        stats["full_rag_messages"] = sum(1 for msg in rag_messages if msg.get("rag_mode") == "full")
        stats["partial_rag_messages"] = sum(1 for msg in rag_messages if msg.get("rag_mode") == "partial")
        stats["fallback_messages"] = sum(1 for msg in rag_messages if msg.get("rag_mode") == "fallback")
        
        # Average processing time for RAG messages
        processing_times = [msg.get("processing_time", 0) for msg in rag_messages if msg.get("processing_time")]
        if processing_times:
            stats["avg_rag_processing_time"] = sum(processing_times) / len(processing_times)
    
    return stats