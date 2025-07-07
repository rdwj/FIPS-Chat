# Chat Guide 05: User Interface Components

## Objective
Create user interface components for document upload, management, and RAG-enhanced chat interactions. This builds the user-facing features that make RAG functionality accessible and intuitive.

## Prerequisites
- Chat Guide 01-04 completed (full RAG pipeline working)
- RAG pipeline processes queries and returns responses with citations
- Document processing and storage systems functional
- Feature branch: `feature/rag-implementation`

## Goals for This Chat Session
1. Create document upload and management interface
2. Enhance chat interface for RAG interactions
3. Build document library and status displays
4. Add RAG controls and configuration options
5. Implement source citation displays and document previews

## Tasks to Complete

### 1. Document Upload Interface
**What to Ask For:**
```
Create ui_components/document_interface.py:
- Document upload component with drag-and-drop support
- File validation (PDF only, size limits)
- Upload progress indicators and status feedback
- Batch upload support for multiple PDFs
- Error handling and user feedback for upload failures
```

### 2. Document Management Dashboard
**What to Ask For:**
```
Add document library features to document_interface.py:
- List view of uploaded documents with metadata
- Document status indicators (processing, ready, error)
- Document statistics (pages, chunks, upload date)
- Delete documents functionality with confirmation
- Search/filter documents by name or date
```

### 3. RAG-Enhanced Chat Interface
**What to Ask For:**
```
Create ui_components/rag_chat_interface.py:
- RAG mode toggle in chat interface
- Source citation display below responses
- Context snippets in expandable sections
- RAG vs. standard mode indicators
- Clear differentiation of RAG-enhanced responses
```

### 4. Source Citation Components
**What to Ask For:**
```
Add source citation and reference features:
- Citation cards showing document, page, section
- Clickable citations that show source excerpts
- Relevance score indicators for sources
- Source snippet highlighting and formatting
- "View full document context" functionality
```

### 5. RAG Configuration Panel
**What to Ask For:**
```
Create RAG settings and configuration interface:
- RAG mode toggle (enable/disable)
- Search sensitivity controls (relevance threshold)
- Number of sources to include setting
- Context length preferences
- RAG statistics and usage metrics
```

### 6. Document Preview and Context
**What to Ask For:**
```
Implement document preview capabilities:
- Quick preview of document content
- Page navigation within documents
- Highlight search terms in document text
- Show chunk boundaries and metadata
- Full-text search within documents
```

### 7. Integration with Main App
**What to Ask For:**
```
Update app.py to include RAG components:
- Add "Documents" tab to main interface
- Integrate RAG chat interface with existing chat
- Add RAG status indicators to sidebar
- Include document upload in sidebar or main area
- Handle RAG-specific session state
```

## Expected Outputs After This Chat
- [ ] `ui_components/document_interface.py` with upload and management
- [ ] `ui_components/rag_chat_interface.py` with enhanced chat features
- [ ] Source citation display components
- [ ] RAG configuration and settings panel
- [ ] Document preview and context features
- [ ] Integration with main application interface
- [ ] Comprehensive UI tests for all components

## Key Implementation Details

### Document Upload Interface
```python
def render_document_upload():
    """Render document upload interface"""
    st.subheader("📄 Upload Documents")
    
    # File uploader with validation
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload PDF documents for RAG enhancement"
    )
    
    if uploaded_files:
        process_uploaded_files(uploaded_files)
    
    # Upload status and progress
    render_upload_status()
    
def process_uploaded_files(files):
    """Process uploaded files with progress tracking"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file in enumerate(files):
        status_text.text(f"Processing {file.name}...")
        # Process file through RAG pipeline
        progress_bar.progress((i + 1) / len(files))
```

### Document Library Display
```python
def render_document_library():
    """Display uploaded documents with management options"""
    st.subheader("📚 Document Library")
    
    # Get documents from storage
    documents = get_document_list()
    
    if not documents:
        st.info("No documents uploaded yet.")
        return
    
    # Document table with actions
    for doc in documents:
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        
        with col1:
            st.write(f"**{doc['filename']}**")
            st.caption(f"{doc['pages']} pages • {doc['chunks']} chunks")
            
        with col2:
            status_badge(doc['status'])
            
        with col3:
            if st.button("👁️", key=f"view_{doc['id']}"):
                show_document_preview(doc['id'])
                
        with col4:
            if st.button("🗑️", key=f"delete_{doc['id']}"):
                delete_document_with_confirmation(doc['id'])
```

### RAG Chat Interface Enhancement
```python
def render_rag_chat_message(message, sources=None):
    """Render chat message with RAG enhancements"""
    # Regular message content
    st.write(message["content"])
    
    # Show timestamp and type
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(f"*{message['timestamp']}*")
    with col2:
        if message.get('rag_mode'):
            st.caption("🔍 RAG Enhanced")
    
    # Source citations if available
    if sources:
        render_source_citations(sources)

def render_source_citations(sources):
    """Display source citations for RAG responses"""
    with st.expander("📚 Sources", expanded=False):
        for i, source in enumerate(sources):
            st.write(f"**{i+1}. {source['document']}** (Page {source['page']})")
            if source.get('section'):
                st.caption(f"Section: {source['section']}")
            
            # Source excerpt
            st.markdown(f"> {source['excerpt']}")
            
            # Relevance score
            st.progress(source['relevance_score'])
            st.caption(f"Relevance: {source['relevance_score']:.2f}")
```

### RAG Configuration Panel
```python
def render_rag_settings():
    """Render RAG configuration options"""
    st.subheader("🔍 RAG Settings")
    
    # Enable/disable RAG
    rag_enabled = st.checkbox(
        "Enable RAG Enhancement",
        value=st.session_state.get("rag_enabled", True),
        help="Use uploaded documents to enhance responses"
    )
    st.session_state.rag_enabled = rag_enabled
    
    if rag_enabled:
        # Search sensitivity
        relevance_threshold = st.slider(
            "Search Sensitivity",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get("rag_threshold", 0.1),
            step=0.05,
            help="Lower values include more sources"
        )
        st.session_state.rag_threshold = relevance_threshold
        
        # Number of sources
        max_sources = st.selectbox(
            "Maximum Sources",
            options=[3, 5, 7, 10],
            index=1,  # Default to 5
            help="Maximum number of sources to include"
        )
        st.session_state.rag_max_sources = max_sources
        
        # Context length
        context_length = st.selectbox(
            "Context Length",
            options=[2000, 3000, 4000, 5000],
            index=1,  # Default to 3000
            help="Maximum context tokens to include"
        )
        st.session_state.rag_context_length = context_length
```

### Document Preview Component
```python
def show_document_preview(doc_id):
    """Show document preview in modal or sidebar"""
    doc = load_document_metadata(doc_id)
    
    st.subheader(f"📄 {doc['filename']}")
    
    # Document statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Pages", doc['total_pages'])
    with col2:
        st.metric("Chunks", doc['total_chunks'])
    with col3:
        st.metric("Size", format_file_size(doc['file_size']))
    
    # Page navigation
    if doc['total_pages'] > 1:
        page_num = st.selectbox(
            "Page",
            options=range(1, doc['total_pages'] + 1),
            index=0
        )
    else:
        page_num = 1
    
    # Show page content
    chunks = get_chunks_for_page(doc_id, page_num)
    for chunk in chunks:
        st.text_area(
            f"Chunk {chunk['chunk_id']}",
            value=chunk['text'],
            height=200,
            disabled=True
        )
```

### Main App Integration
```python
# Addition to app.py
def render_main_content():
    """Enhanced main content with RAG tab"""
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "🖼️ Image Analysis", "📄 Documents", "🔧 Models"])
    
    with tab1:
        render_chat_tab()
    
    with tab2:
        render_image_tab()
    
    with tab3:
        render_documents_tab()  # New RAG tab
    
    with tab4:
        render_models_tab()

def render_documents_tab():
    """Render RAG documents tab"""
    # Check if RAG is available
    if not rag_system_available():
        st.warning("RAG system not configured. Please check deployment.")
        return
    
    # Document upload and management
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_document_upload()
        render_document_library()
    
    with col2:
        render_rag_settings()
        render_rag_statistics()
```

## User Experience Requirements

### Upload Experience
- Clear drag-and-drop area with visual feedback
- Progress indicators for file processing
- Error messages with actionable guidance
- Batch upload support for multiple files
- File validation with helpful error messages

### Chat Experience
- Clear indication when RAG is active
- Source citations that don't overwhelm the response
- Easy switching between RAG and standard modes
- Visual differentiation of RAG-enhanced responses
- Quick access to source documents

### Document Management
- Easy document discovery and navigation
- Clear status indicators for processing
- Simple deletion with confirmation
- Document search and filtering
- Storage usage indicators

## Success Criteria for This Chat
- ✅ Document upload works smoothly with good UX
- ✅ Document library shows all uploaded files with status
- ✅ RAG chat interface clearly shows enhanced responses
- ✅ Source citations are informative and accessible
- ✅ RAG settings are intuitive and functional
- ✅ Integration with main app is seamless

## Accessibility and Usability
- Clear labels and help text for all controls
- Keyboard navigation support
- Screen reader friendly components
- Mobile-responsive design
- Error states with recovery guidance

## Next Chat Guide
After completing the UI components, proceed to **Chat Guide 06: Configuration & Settings** to add environment variables, configuration management, and deployment settings.

## Notes for Implementation
- Follow existing Streamlit patterns from the app
- Maintain consistent styling with existing interface
- Test all components with realistic document sets
- Ensure error handling provides helpful user feedback
- Consider mobile and tablet usage scenarios
- Make RAG features discoverable but not overwhelming