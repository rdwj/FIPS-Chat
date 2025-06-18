# RAG (Retrieval-Augmented Generation) Features

## Overview

FIPS Chat now includes a comprehensive RAG system that enhances AI responses with relevant information from your uploaded documents. The system provides document upload, processing, search, and integration with AI chat for context-aware responses.

## ✨ Features

### 📄 Document Management
- **PDF Upload**: Drag-and-drop PDF upload with validation
- **Document Library**: Browse, search, and manage uploaded documents
- **Document Preview**: View document content, pages, and chunks
- **Storage Management**: Monitor usage and clean up old documents
- **Batch Processing**: Upload multiple documents simultaneously

### 🔍 RAG-Enhanced Chat
- **Context-Aware Responses**: AI responses enhanced with relevant document excerpts
- **Source Citations**: Transparent citations showing which documents were used
- **Adaptive Quality**: Automatic quality assessment with fallback modes
- **Real-time Search**: Intelligent document search during conversations
- **Processing Statistics**: Track RAG effectiveness and performance

### ⚙️ Configuration & Settings
- **Chunking Controls**: Adjust text processing parameters
- **Search Sensitivity**: Configure relevance thresholds
- **Context Length**: Control how much context to include
- **Memory Management**: Optimize memory usage for document caching

## 🚀 Getting Started

### 1. Install Dependencies

The RAG system requires additional dependencies that are included in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `docling>=1.0.0` - PDF processing and text extraction
- `scikit-learn>=1.3.0` - TF-IDF search functionality
- `scipy>=1.10.0` - Scientific computing for search
- `numpy>=1.24.0` - Numerical operations
- `psutil>=5.9.0` - Memory monitoring

### 2. Configure API Endpoint

1. Start the application: `streamlit run app.py`
2. Configure your API endpoint in the sidebar
3. Test the connection and discover models
4. Select a chat model for RAG enhancement

### 3. Upload Documents

1. Go to the **Documents** tab
2. Use the **Upload** sub-tab to add PDF documents
3. Configure upload settings (overwrite, auto-index)
4. Click **Process Documents** to upload and process files
5. Monitor processing progress and results

### 4. Use RAG Chat

1. Switch to the **RAG Chat** tab
2. Enable RAG enhancement using the checkbox
3. Configure RAG settings (sensitivity, sources, context)
4. Start chatting - responses will include relevant document context
5. View source citations and excerpts for transparency

## 🎛️ Interface Guide

### Documents Tab

#### Upload Sub-tab
- **File Uploader**: Drag-and-drop area for PDF files
- **Upload Settings**: 
  - Overwrite existing documents
  - Auto-build search index after upload
- **Progress Tracking**: Real-time processing status
- **Results Summary**: Success/failure counts

#### Library Sub-tab
- **Document List**: All uploaded documents with metadata
- **Filters**: Search by filename, filter by status
- **Sort Options**: By date, filename, pages, or chunks
- **Actions**: Preview, statistics, delete for each document
- **Document Preview**: View pages and chunks within documents

#### Settings Sub-tab
- **Chunking Configuration**: Adjust chunk size and overlap
- **Storage Settings**: Memory limits and cleanup options
- **System Actions**: Rebuild index, view statistics, clean storage

### RAG Chat Tab

#### Chat Interface
- **RAG Toggle**: Enable/disable RAG enhancement
- **Status Indicator**: Shows RAG availability and document count
- **Message History**: Enhanced messages with source citations
- **Source Citations**: Expandable sections showing document excerpts
- **Processing Times**: Transparent performance metrics

#### RAG Controls
- **Settings Panel**: Configure search sensitivity and context
- **Statistics Panel**: View RAG effectiveness metrics
- **Mode Indicators**: Visual feedback on RAG, partial, or fallback responses

## 📊 Understanding RAG Modes

The system automatically adapts based on content quality:

### 🔍 Full RAG Mode
- High-quality, relevant context found
- Response fully informed by documents
- Multiple source citations provided
- Highest confidence in accuracy

### 🔎 Partial RAG Mode  
- Some relevant context found
- Supplements with general knowledge
- Clear distinction between sources
- Moderate confidence level

### 💬 Fallback Mode
- No relevant context found
- Uses general AI knowledge
- Clearly indicated as non-document based
- Suggests what documents might help

## ⚙️ Configuration Options

### Document Processing
```python
RAGConfig(
    chunk_size=1000,        # Characters per chunk
    chunk_overlap=200,      # Overlap between chunks
    min_chunk_size=100,     # Minimum chunk size
    max_chunk_size=2000     # Maximum chunk size
)
```

### Search Parameters
- **Relevance Threshold**: 0.0-1.0 (lower = more inclusive)
- **Maximum Sources**: 3-10 sources per response
- **Context Length**: 2000-5000 tokens maximum

### Memory Management
- **Document Cache**: 50-500 MB memory limit
- **Auto Cleanup**: Remove old documents automatically
- **Storage Monitoring**: Track disk and memory usage

## 📈 Performance Tips

### Document Optimization
- Use clear, well-structured PDFs
- Avoid image-heavy documents (text extraction works best)
- Break large documents into logical sections
- Use descriptive filenames

### RAG Configuration
- Start with default settings (1000 char chunks, 0.1 threshold)
- Increase chunk overlap for better context continuity
- Lower threshold for broader search, higher for precision
- Monitor statistics to optimize settings

### Memory Management
- Regular cleanup of old documents
- Monitor cache hit rates (aim for >50%)
- Adjust memory limits based on available resources
- Use auto-cleanup for maintenance

## 🔧 Troubleshooting

### Common Issues

#### "RAG system not available"
- Check that all dependencies are installed
- Verify `docling` is properly installed
- Ensure sufficient disk space for storage

#### "No relevant context found"
- Check document upload was successful
- Verify search index was built
- Lower relevance threshold in settings
- Try different query phrasing

#### "Processing failed"
- Ensure PDF is not password-protected
- Check file is valid PDF format
- Verify sufficient memory available
- Check application logs for details

#### Poor RAG performance
- Review document quality and structure
- Adjust chunk size for your content type
- Fine-tune relevance threshold
- Monitor processing statistics

### Performance Issues
- **Slow uploads**: Check available memory and disk space
- **Slow search**: Rebuild search index, reduce document count
- **High memory usage**: Reduce cache size, clean old documents
- **Poor relevance**: Adjust chunk size and overlap settings

## 🧪 Testing & Development

### Example Usage
Run the example script to test RAG functionality:

```bash
python rag/example_rag_usage.py
```

### Manual Testing
1. Upload a test PDF document
2. Verify processing completes successfully
3. Test search functionality in RAG chat
4. Check source citations are accurate
5. Verify statistics and monitoring

### Storage Validation
Use the storage validation feature to check integrity:
- Go to Documents → Settings
- Click "Storage Statistics"
- Review validation results
- Fix any reported issues

## 📚 Technical Architecture

### Components
- **DoclingProcessor**: PDF processing and text chunking
- **FileStorage**: Memory-efficient document storage
- **TFIDFSearchEngine**: Fast text search and ranking
- **RAGPipeline**: End-to-end response generation
- **ContextManager**: Intelligent context selection
- **QualityController**: Response quality assessment

### Data Flow
1. **Upload**: PDF → DoclingProcessor → DocumentResult
2. **Storage**: DocumentResult → FileStorage → Disk/Memory
3. **Indexing**: Documents → TFIDFSearchEngine → Search Index
4. **Query**: User Input → Search → Context → AI → Enhanced Response
5. **Citation**: Sources → UI → User Transparency

### Security Features
- FIPS-compliant SHA-256 hashing for document IDs
- Secure temporary file handling
- Memory-safe document processing
- Input validation and sanitization

## 🤝 Contributing

When contributing to RAG features:

1. **Test thoroughly** with various document types
2. **Monitor performance** impact on memory and processing
3. **Update documentation** for new features
4. **Follow security guidelines** for file handling
5. **Maintain backward compatibility** with existing features

## 📋 Changelog

### Version 1.0 (Current)
- ✅ Complete RAG system implementation
- ✅ Document upload and management UI
- ✅ RAG-enhanced chat interface
- ✅ Source citation and transparency
- ✅ Adaptive quality control
- ✅ Performance monitoring and statistics
- ✅ Memory-efficient storage system
- ✅ FIPS-compliant security features

### Future Enhancements
- 🔄 Support for additional document formats (DOCX, TXT, HTML)
- 🔄 Advanced semantic search with embeddings
- 🔄 Document summarization and key extraction
- 🔄 Multi-language document support
- 🔄 Integration with external document sources
- 🔄 Advanced analytics and insights dashboard

---

For technical support or questions about RAG features, please refer to the main README.md or create an issue in the project repository.