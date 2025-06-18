# Chat Guide 01: Document Processing Foundation

## Objective
Build the foundation for document processing using Docling while ensuring FIPS compliance. This chat session will establish the core document extraction and chunking capabilities.

## Prerequisites
- Feature branch created: `feature/rag-implementation`
- Current working directory: `/Users/wjackson/Developer/FIPS-Chat`

## Goals for This Chat Session
1. Add Docling dependencies and validate FIPS compliance
2. Create basic document processing module
3. Implement FIPS-safe PDF text extraction
4. Build intelligent text chunking with metadata preservation
5. Create initial tests to validate functionality

## Tasks to Complete

### 1. Dependency Management
**What to Ask For:**
```
I need to add Docling dependencies to requirements.txt for PDF processing. 
Add these dependencies and verify they are FIPS-compliant:
- docling>=1.0.0
- docling-core>=1.0.0
- docling-parse>=1.0.0
- scikit-learn>=1.3.0 (for TF-IDF later)
- numpy>=1.24.0

Test that these can be imported in a FIPS environment without triggering MD5 issues.
```

### 2. RAG Module Structure
**What to Ask For:**
```
Create the basic RAG module structure:
- Create rag/ directory with __init__.py
- Create rag/document_processor.py with DoclingProcessor class
- Set up error handling and logging for document processing
- Create basic configuration for document processing parameters
```

### 3. PDF Text Extraction
**What to Ask For:**
```
Implement PDF text extraction using Docling in rag/document_processor.py:
- DoclingProcessor class with process_pdf() method
- Extract text, tables, and document structure
- Preserve page numbers and section headings
- Use FIPS-compliant hashing (SHA-256) for document fingerprints
- Handle errors gracefully (corrupted PDFs, unsupported formats)
```

### 4. Intelligent Text Chunking
**What to Ask For:**
```
Add text chunking functionality to DoclingProcessor:
- Respect document structure (headings, paragraphs, tables)
- Configurable chunk size (default 1000 chars) with overlap (default 200 chars)
- Preserve metadata for each chunk (page number, section title, position)
- Handle special cases (tables, lists, code blocks)
- Ensure chunks don't break mid-sentence when possible
```

### 5. Metadata Management
**What to Ask For:**
```
Create comprehensive metadata structure for documents and chunks:
- Document-level metadata (filename, upload time, total pages, processing status)
- Chunk-level metadata (chunk ID, page number, section, character positions)
- FIPS-compliant hashing for unique IDs
- JSON serialization for metadata storage
```

### 6. Basic Testing
**What to Ask For:**
```
Create tests/test_document_processing.py:
- Test PDF text extraction with sample PDF
- Test chunking with various document structures
- Test metadata generation and serialization
- Test FIPS compliance (no MD5 usage)
- Test error handling for invalid files
```

## Expected Outputs After This Chat
- [ ] `requirements.txt` updated with RAG dependencies
- [ ] `rag/` module created with proper structure
- [ ] `rag/document_processor.py` with DoclingProcessor class
- [ ] Text extraction and chunking functionality working
- [ ] Comprehensive metadata structure defined
- [ ] Basic tests validating core functionality
- [ ] FIPS compliance verified for all document processing

## Key Implementation Details

### Document Processing Class Structure
```python
class DoclingProcessor:
    def __init__(self, config: RAGConfig):
        # Initialize Docling converter
        # Set up FIPS-compliant hashing
        
    def process_pdf(self, file_path: str) -> DocumentResult:
        # Extract text, tables, metadata using Docling
        # Create chunks with structure preservation
        # Generate FIPS-compliant document hash
        
    def create_chunks(self, docling_result, chunk_size=1000, overlap=200):
        # Intelligent chunking with structure awareness
        # Preserve metadata for each chunk
```

### Metadata Structure
```python
{
    "doc_id": "sha256_hash",
    "filename": "document.pdf",
    "upload_timestamp": "2024-01-01T00:00:00Z",
    "total_pages": 15,
    "total_chunks": 23,
    "processing_status": "completed",
    "chunks": [
        {
            "chunk_id": "chunk_001", 
            "text": "chunk content...",
            "page_number": 1,
            "section_title": "Introduction",
            "char_start": 0,
            "char_end": 1000,
            "word_count": 187
        }
    ]
}
```

## Success Criteria for This Chat
- ✅ Docling successfully extracts text from PDF without FIPS violations
- ✅ Text chunking preserves document structure and metadata
- ✅ All components use FIPS-compliant hashing (SHA-256 only)
- ✅ Tests pass and validate core functionality
- ✅ Code is clean, documented, and follows project patterns

## Next Chat Guide
After completing this foundation, proceed to **Chat Guide 02: File-Based Storage System** to implement the storage layer for documents and chunks.

## Notes for Implementation
- Test every component in FIPS mode before proceeding
- Use SHA-256 for all hashing operations
- Monitor memory usage during document processing
- Ensure graceful error handling for edge cases
- Follow existing project code style and patterns