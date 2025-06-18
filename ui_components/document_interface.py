"""Document interface component for RAG document upload and management."""

import streamlit as st
import tempfile
import os
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import time

from rag.document_processor import DoclingProcessor, RAGConfig
from rag.file_storage import FileStorage, DocumentIndex
from rag.search_engine import TFIDFSearchEngine

logger = logging.getLogger(__name__)


def initialize_rag_system():
    """Initialize RAG system components if not already done."""
    if "rag_storage" not in st.session_state:
        st.session_state.rag_storage = FileStorage("rag_storage", max_memory_mb=150)
    
    if "rag_processor" not in st.session_state:
        config = RAGConfig(
            chunk_size=st.session_state.get("rag_chunk_size", 1000),
            chunk_overlap=st.session_state.get("rag_chunk_overlap", 200),
            min_chunk_size=100,
            max_chunk_size=2000
        )
        st.session_state.rag_processor = DoclingProcessor(
            config=config, 
            storage=st.session_state.rag_storage,
            auto_store=True
        )
    
    if "search_engine" not in st.session_state:
        st.session_state.search_engine = TFIDFSearchEngine(st.session_state.rag_storage)
        # Initialize search engine from stored documents
        try:
            st.session_state.search_engine.build_index_from_storage()
        except Exception as e:
            logger.error(f"Error building search index: {e}")


def check_rag_system_available() -> bool:
    """Check if RAG system is available and working."""
    try:
        initialize_rag_system()
        return True
    except Exception as e:
        logger.error(f"RAG system unavailable: {e}")
        return False


def render_document_interface():
    """Main document interface rendering function."""
    st.header("📄 Document Management")
    
    # Check RAG system availability
    if not check_rag_system_available():
        st.error("⚠️ RAG system is not available. Please check your configuration.")
        st.info("The RAG system requires proper dependencies and configuration to function.")
        return
    
    # Create tabs for different document operations
    tab1, tab2, tab3 = st.tabs(["📤 Upload", "📚 Library", "⚙️ Settings"])
    
    with tab1:
        render_document_upload()
    
    with tab2:
        render_document_library()
    
    with tab3:
        render_document_settings()


def render_document_upload():
    """Render document upload interface with drag-and-drop support."""
    st.subheader("📤 Upload Documents")
    
    # File uploader with validation
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload PDF documents for RAG enhancement. Files will be processed and indexed for search.",
        key="document_uploader"
    )
    
    # Upload configuration
    col1, col2 = st.columns(2)
    with col1:
        overwrite_existing = st.checkbox(
            "Overwrite existing documents",
            value=False,
            help="Replace documents with same content hash"
        )
    
    with col2:
        auto_index = st.checkbox(
            "Auto-build search index",
            value=True,
            help="Automatically rebuild search index after upload"
        )
    
    if uploaded_files:
        if st.button("🚀 Process Documents", type="primary"):
            process_uploaded_files(uploaded_files, overwrite_existing, auto_index)
    
    # Show recent upload status
    render_upload_status()


def process_uploaded_files(files: List, overwrite_existing: bool = False, auto_index: bool = True):
    """Process uploaded files with progress tracking."""
    if not files:
        return
    
    total_files = len(files)
    progress_bar = st.progress(0)
    status_container = st.container()
    
    processed_count = 0
    failed_count = 0
    skipped_count = 0
    
    for i, uploaded_file in enumerate(files):
        with status_container:
            st.write(f"📄 Processing **{uploaded_file.name}** ({i+1}/{total_files})")
        
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Check if document already exists
                file_content = uploaded_file.getvalue()
                existing_doc_id = st.session_state.rag_processor.check_document_exists(
                    file_content.decode('utf-8', errors='ignore')[:1000]  # Sample for checking
                )
                
                if existing_doc_id and not overwrite_existing:
                    with status_container:
                        st.warning(f"⚠️ Document **{uploaded_file.name}** already exists. Skipping.")
                    skipped_count += 1
                else:
                    # Process the document
                    with status_container:
                        processing_spinner = st.spinner(f"Processing {uploaded_file.name}...")
                    
                    with processing_spinner:
                        doc_result = st.session_state.rag_processor.process_pdf(tmp_file_path)
                        doc_id = st.session_state.rag_storage.store_document(doc_result)
                    
                    with status_container:
                        st.success(f"✅ **{uploaded_file.name}** processed successfully!")
                        st.caption(f"Document ID: {doc_id[:12]}... | {doc_result.metadata.total_chunks} chunks created")
                    
                    processed_count += 1
                    
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_file_path)
                except Exception:
                    pass
        
        except Exception as e:
            error_msg = str(e)
            with status_container:
                st.error(f"❌ Failed to process **{uploaded_file.name}**: {error_msg}")
            failed_count += 1
            logger.error(f"Failed to process {uploaded_file.name}: {error_msg}")
        
        # Update progress
        progress_bar.progress((i + 1) / total_files)
    
    # Final status
    with status_container:
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("✅ Processed", processed_count)
        with col2:
            st.metric("⚠️ Skipped", skipped_count)
        with col3:
            st.metric("❌ Failed", failed_count)
    
    # Rebuild search index if requested and documents were processed
    if auto_index and processed_count > 0:
        with st.spinner("Building search index..."):
            try:
                st.session_state.search_engine.build_index_from_storage()
                st.success("🔍 Search index updated successfully!")
            except Exception as e:
                st.error(f"Failed to update search index: {str(e)}")
    
    # Clear file uploader
    if processed_count > 0:
        st.rerun()


def render_upload_status():
    """Render upload status and recent activity."""
    if "upload_history" not in st.session_state:
        st.session_state.upload_history = []
    
    if st.session_state.upload_history:
        with st.expander("📊 Recent Upload Activity", expanded=False):
            for entry in st.session_state.upload_history[-5:]:  # Show last 5
                st.write(f"**{entry['filename']}** - {entry['status']} - {entry['timestamp']}")


def render_document_library():
    """Display uploaded documents with management options."""
    st.subheader("📚 Document Library")
    
    # Get documents from storage
    try:
        documents = st.session_state.rag_storage.list_documents()
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return
    
    if not documents:
        st.info("📭 No documents uploaded yet. Use the Upload tab to add PDF documents.")
        return
    
    # Document filters
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox(
            "Status Filter",
            options=["All", "completed", "processing", "error"],
            index=0
        )
    
    with col2:
        search_query = st.text_input(
            "Search documents",
            placeholder="Search by filename...",
            key="doc_search"
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort by",
            options=["Upload Date", "Filename", "Pages", "Chunks"],
            index=0
        )
    
    # Apply filters
    filtered_docs = documents
    if status_filter != "All":
        filtered_docs = [d for d in filtered_docs if d.status == status_filter]
    
    if search_query:
        filtered_docs = [d for d in filtered_docs if search_query.lower() in d.filename.lower()]
    
    # Sort documents
    if sort_by == "Filename":
        filtered_docs.sort(key=lambda x: x.filename.lower())
    elif sort_by == "Pages":
        filtered_docs.sort(key=lambda x: x.total_pages, reverse=True)
    elif sort_by == "Chunks":
        filtered_docs.sort(key=lambda x: x.total_chunks, reverse=True)
    else:  # Upload Date
        filtered_docs.sort(key=lambda x: x.upload_date, reverse=True)
    
    # Display document count
    st.write(f"**{len(filtered_docs)}** documents {f'(filtered from {len(documents)})' if len(filtered_docs) != len(documents) else ''}")
    
    # Document list with actions
    for doc in filtered_docs:
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([4, 1, 1, 1, 1])
            
            with col1:
                st.write(f"**📄 {doc.filename}**")
                
                # Document stats
                stats_text = f"{doc.total_pages} pages • {doc.total_chunks} chunks"
                if hasattr(doc, 'file_size_bytes') and doc.file_size_bytes:
                    size_mb = doc.file_size_bytes / (1024 * 1024)
                    stats_text += f" • {size_mb:.1f} MB"
                
                st.caption(stats_text)
                
                # Upload date
                try:
                    upload_date = datetime.fromisoformat(doc.upload_date)
                    st.caption(f"Uploaded: {upload_date.strftime('%Y-%m-%d %H:%M')}")
                except Exception:
                    st.caption(f"Uploaded: {doc.upload_date}")
            
            with col2:
                render_status_badge(doc.status)
            
            with col3:
                if st.button("👁️", key=f"view_{doc.doc_id}", help="Preview document"):
                    show_document_preview(doc)
            
            with col4:
                if st.button("📊", key=f"stats_{doc.doc_id}", help="Document statistics"):
                    show_document_stats(doc)
            
            with col5:
                if st.button("🗑️", key=f"delete_{doc.doc_id}", help="Delete document"):
                    confirm_document_deletion(doc)
        
        st.divider()


def render_status_badge(status: str):
    """Render a status badge."""
    if status == "completed":
        st.success("✅ Ready")
    elif status == "processing":
        st.info("⏳ Processing")
    elif status == "error":
        st.error("❌ Error")
    else:
        st.warning(f"❓ {status}")


def show_document_preview(doc: DocumentIndex):
    """Show document preview in an expander."""
    with st.expander(f"📄 Preview: {doc.filename}", expanded=True):
        try:
            # Load document chunks
            doc_result = st.session_state.rag_storage.load_document(doc.doc_id)
            
            if not doc_result:
                st.error("Failed to load document content")
                return
            
            # Document metadata
            st.subheader("Document Information")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Pages", doc.total_pages)
            with col2:
                st.metric("Chunks", doc.total_chunks)
            with col3:
                if hasattr(doc, 'file_size_bytes'):
                    size_mb = doc.file_size_bytes / (1024 * 1024)
                    st.metric("Size", f"{size_mb:.1f} MB")
            with col4:
                st.metric("Status", doc.status)
            
            # Page navigation if multiple pages
            if doc.total_pages > 1:
                selected_page = st.selectbox(
                    "Select Page",
                    options=list(range(1, doc.total_pages + 1)),
                    index=0,
                    key=f"page_select_{doc.doc_id}"
                )
            else:
                selected_page = 1
            
            # Show chunks for selected page
            page_chunks = [chunk for chunk in doc_result.chunks 
                          if chunk.metadata.page_number == selected_page]
            
            if page_chunks:
                st.subheader(f"Page {selected_page} Content")
                for i, chunk in enumerate(page_chunks):
                    with st.expander(f"Chunk {i+1} ({chunk.metadata.word_count} words)", expanded=False):
                        st.text_area(
                            "Content",
                            value=chunk.text,
                            height=200,
                            disabled=True,
                            key=f"chunk_content_{doc.doc_id}_{i}"
                        )
                        
                        # Chunk metadata
                        st.caption(f"Section: {chunk.metadata.section_title}")
                        st.caption(f"Position: {chunk.metadata.char_start}-{chunk.metadata.char_end}")
            else:
                st.warning(f"No content found for page {selected_page}")
                
        except Exception as e:
            st.error(f"Error loading document preview: {str(e)}")


def show_document_stats(doc: DocumentIndex):
    """Show detailed document statistics."""
    with st.expander(f"📊 Statistics: {doc.filename}", expanded=True):
        try:
            # Load document for analysis
            doc_result = st.session_state.rag_storage.load_document(doc.doc_id)
            
            if not doc_result:
                st.error("Failed to load document for analysis")
                return
            
            # Basic stats
            st.subheader("Document Overview")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Pages", doc.total_pages)
                st.metric("Total Chunks", doc.total_chunks)
                if hasattr(doc, 'file_size_bytes'):
                    st.metric("File Size (MB)", f"{doc.file_size_bytes / (1024*1024):.1f}")
            
            with col2:
                # Calculate text statistics
                total_words = sum(chunk.metadata.word_count for chunk in doc_result.chunks)
                total_chars = sum(len(chunk.text) for chunk in doc_result.chunks)
                avg_chunk_size = total_words / len(doc_result.chunks) if doc_result.chunks else 0
                
                st.metric("Total Words", f"{total_words:,}")
                st.metric("Total Characters", f"{total_chars:,}")
                st.metric("Avg Chunk Size", f"{avg_chunk_size:.0f} words")
            
            # Chunks per page distribution
            st.subheader("Content Distribution")
            page_chunks = {}
            for chunk in doc_result.chunks:
                page = chunk.metadata.page_number
                page_chunks[page] = page_chunks.get(page, 0) + 1
            
            if page_chunks:
                st.bar_chart(page_chunks)
            
            # Word count distribution
            st.subheader("Chunk Size Distribution")
            word_counts = [chunk.metadata.word_count for chunk in doc_result.chunks]
            if word_counts:
                import pandas as pd
                df = pd.DataFrame({'Word Count': word_counts})
                st.histogram(df, x='Word Count', bins=20)
                
        except Exception as e:
            st.error(f"Error generating statistics: {str(e)}")


def confirm_document_deletion(doc: DocumentIndex):
    """Show document deletion confirmation dialog."""
    st.warning(f"⚠️ Delete **{doc.filename}**?")
    st.write("This action cannot be undone. The document and all its chunks will be permanently removed.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Yes, Delete", type="primary", key=f"confirm_delete_{doc.doc_id}"):
            delete_document(doc)
    
    with col2:
        if st.button("❌ Cancel", key=f"cancel_delete_{doc.doc_id}"):
            st.rerun()


def delete_document(doc: DocumentIndex):
    """Delete a document and update the search index."""
    try:
        success = st.session_state.rag_storage.delete_document(doc.doc_id)
        
        if success:
            st.success(f"✅ **{doc.filename}** deleted successfully!")
            
            # Rebuild search index
            with st.spinner("Updating search index..."):
                st.session_state.search_engine.build_index_from_storage()
            
            # Add to history
            if "upload_history" not in st.session_state:
                st.session_state.upload_history = []
            
            st.session_state.upload_history.append({
                'filename': doc.filename,
                'status': 'deleted',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            st.rerun()
        else:
            st.error(f"❌ Failed to delete **{doc.filename}**")
            
    except Exception as e:
        st.error(f"Error deleting document: {str(e)}")


def render_document_settings():
    """Render document processing settings."""
    st.subheader("⚙️ Document Processing Settings")
    
    # Chunking settings
    st.write("**Text Chunking Configuration**")
    
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.slider(
            "Chunk Size (characters)",
            min_value=500,
            max_value=3000,
            value=st.session_state.get("rag_chunk_size", 1000),
            step=100,
            help="Size of text chunks for processing"
        )
        st.session_state.rag_chunk_size = chunk_size
    
    with col2:
        chunk_overlap = st.slider(
            "Chunk Overlap (characters)",
            min_value=0,
            max_value=500,
            value=st.session_state.get("rag_chunk_overlap", 200),
            step=50,
            help="Overlap between consecutive chunks"
        )
        st.session_state.rag_chunk_overlap = chunk_overlap
    
    # Storage settings
    st.divider()
    st.write("**Storage Configuration**")
    
    col1, col2 = st.columns(2)
    with col1:
        max_memory_mb = st.slider(
            "Memory Cache (MB)",
            min_value=50,
            max_value=500,
            value=150,
            step=25,
            help="Maximum memory for document caching"
        )
    
    with col2:
        auto_cleanup = st.checkbox(
            "Auto Cleanup",
            value=st.session_state.get("rag_auto_cleanup", False),
            help="Automatically clean up old documents"
        )
        st.session_state.rag_auto_cleanup = auto_cleanup
    
    # System actions
    st.divider()
    st.write("**System Management**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Rebuild Search Index"):
            rebuild_search_index()
    
    with col2:
        if st.button("📊 Storage Statistics"):
            show_storage_statistics()
    
    with col3:
        if st.button("🧹 Clean Storage"):
            clean_storage()


def rebuild_search_index():
    """Rebuild the search index from stored documents."""
    try:
        with st.spinner("Rebuilding search index..."):
            st.session_state.search_engine.build_index_from_storage()
        st.success("✅ Search index rebuilt successfully!")
        
        # Show index stats
        stats = st.session_state.search_engine.get_index_stats()
        st.info(f"Indexed {stats.get('total_documents', 0)} documents with {stats.get('total_terms', 0)} unique terms")
        
    except Exception as e:
        st.error(f"Failed to rebuild search index: {str(e)}")


def show_storage_statistics():
    """Display storage system statistics."""
    try:
        stats = st.session_state.rag_storage.get_storage_stats()
        
        with st.expander("📊 Storage Statistics", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Documents", stats.total_documents)
                st.metric("Total Chunks", stats.total_chunks)
            
            with col2:
                disk_mb = stats.total_disk_usage_bytes / (1024 * 1024)
                memory_mb = stats.memory_usage_bytes / (1024 * 1024)
                st.metric("Disk Usage", f"{disk_mb:.1f} MB")
                st.metric("Memory Usage", f"{memory_mb:.1f} MB")
            
            with col3:
                st.metric("Cache Hit Rate", f"{stats.cache_hit_rate:.2%}")
                
        # Search engine stats if available
        try:
            search_stats = st.session_state.search_engine.get_index_stats()
            with st.expander("🔍 Search Index Statistics", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Indexed Documents", search_stats.get('total_documents', 0))
                    st.metric("Unique Terms", search_stats.get('total_terms', 0))
                with col2:
                    st.metric("Index Size", f"{search_stats.get('index_size_mb', 0):.1f} MB")
                    st.metric("Last Updated", search_stats.get('last_updated', 'Never'))
        except Exception as e:
            st.warning(f"Search index stats unavailable: {str(e)}")
        
    except Exception as e:
        st.error(f"Error getting storage statistics: {str(e)}")


def clean_storage():
    """Clean up storage with confirmation."""
    st.warning("⚠️ Storage cleanup will remove old and orphaned files.")
    
    max_age_days = st.slider(
        "Remove documents older than (days)",
        min_value=1,
        max_value=365,
        value=30,
        help="Documents older than this will be removed"
    )
    
    if st.button("🧹 Confirm Cleanup", type="primary"):
        try:
            with st.spinner("Cleaning storage..."):
                cleanup_results = st.session_state.rag_storage.cleanup_storage(max_age_days)
            
            st.success("✅ Storage cleanup completed!")
            st.write(f"Removed {cleanup_results.get('removed_orphaned_files', 0)} orphaned files")
            st.write(f"Removed {cleanup_results.get('removed_old_documents', 0)} old documents")
            
            # Rebuild search index after cleanup  
            if cleanup_results.get('removed_old_documents', 0) > 0:
                with st.spinner("Updating search index..."):
                    st.session_state.search_engine.build_index_from_storage()
                    
        except Exception as e:
            st.error(f"Error during cleanup: {str(e)}")


def get_document_statistics() -> Dict[str, Any]:
    """Get document management statistics for the main app."""
    try:
        if not check_rag_system_available():
            return {}
        
        # Storage stats
        storage_stats = st.session_state.rag_storage.get_storage_stats()
        
        # Recent activity
        recent_docs = st.session_state.rag_storage.list_documents()
        recent_count = len([d for d in recent_docs if d.upload_date > (datetime.now().isoformat()[:10])])  # Today
        
        return {
            'total_documents': storage_stats.total_documents,
            'total_chunks': storage_stats.total_chunks,
            'disk_usage_mb': storage_stats.total_disk_usage_bytes / (1024 * 1024),
            'memory_usage_mb': storage_stats.memory_usage_bytes / (1024 * 1024),
            'recent_documents': recent_count,
            'cache_hit_rate': storage_stats.cache_hit_rate
        }
        
    except Exception as e:
        logger.error(f"Error getting document statistics: {e}")
        return {}