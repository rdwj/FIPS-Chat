# Chat Guide 02: File-Based Storage System

## Objective
Implement a file-based storage system for documents and chunks that is memory-efficient and FIPS-compliant. This replaces traditional vector databases to avoid MD5 issues.

## Prerequisites
- Chat Guide 01 completed (document processing foundation)
- DoclingProcessor working and FIPS-validated
- Feature branch: `feature/rag-implementation`

## Goals for This Chat Session
1. Design file-based storage structure for documents and chunks
2. Implement document storage with efficient retrieval
3. Create memory-efficient operations for large document sets
4. Build indexing system for fast document lookup
5. Add storage management utilities (cleanup, statistics)

## Tasks to Complete

### 1. Storage Architecture Design
**What to Ask For:**
```
Design and create the file-based storage architecture:
- Create rag_storage/ directory structure (documents/, indexes/, config/)
- Design JSON schema for document and chunk storage
- Plan memory-efficient loading strategies
- Create storage configuration management
```

### 2. File Storage Implementation
**What to Ask For:**
```
Create rag/file_storage.py with FileStorage class:
- store_document() method to save processed documents
- load_document() method with lazy loading for memory efficiency
- delete_document() method with cleanup
- list_documents() method with metadata summary
- get_storage_stats() for monitoring usage
```

### 3. Memory Management
**What to Ask For:**
```
Implement memory-efficient operations in rag/file_storage.py:
- LRU cache for frequently accessed documents
- Lazy loading of document chunks
- Memory usage monitoring and limits
- Automatic cleanup when memory threshold reached
- Stream processing for large documents
```

### 4. Document Indexing
**What to Ask For:**
```
Create indexing system for fast document retrieval:
- Document index by filename, upload date, size
- Chunk index by document ID and page number
- Term frequency index for search preparation
- FIPS-compliant indexing (no MD5 hashing)
- Incremental index updates
```

### 5. Storage Utilities
**What to Ask For:**
```
Add storage management utilities:
- Storage cleanup (remove old/orphaned files)
- Storage statistics (total docs, chunks, disk usage)
- Document validation (check for corruption)
- Backup/export functionality
- Storage migration tools
```

### 6. Integration with Document Processor
**What to Ask For:**
```
Integrate storage with document processing:
- Update DoclingProcessor to save results automatically
- Add storage callbacks to document processing pipeline
- Implement document deduplication based on content hash
- Handle storage errors during processing
```

### 7. Testing
**What to Ask For:**
```
Create tests/test_file_storage.py:
- Test document storage and retrieval
- Test memory management and lazy loading
- Test indexing operations
- Test storage cleanup and utilities
- Test integration with document processor
- Verify FIPS compliance throughout
```

## Expected Outputs After This Chat
- [ ] `rag_storage/` directory structure created
- [ ] `rag/file_storage.py` with FileStorage class
- [ ] Memory-efficient document loading/caching
- [ ] Document and chunk indexing system
- [ ] Storage management utilities
- [ ] Integration with document processing pipeline
- [ ] Comprehensive tests for storage operations

## Key Implementation Details

### Storage Directory Structure
```
rag_storage/
├── documents/           # Document storage
│   ├── doc_abcd1234.json     # Document metadata & chunks
│   ├── doc_efgh5678.json
│   └── ...
├── indexes/            # Search indexes
│   ├── document_index.json   # Document lookup index
│   ├── chunk_index.json      # Chunk-level index
│   └── term_index.json       # Term frequency index
└── config/
    ├── storage_config.json   # Storage settings
    └── stats.json           # Storage statistics
```

### FileStorage Class Interface
```python
class FileStorage:
    def __init__(self, storage_path: str, max_memory_mb: int = 100):
        # Initialize storage, create directories
        # Set up memory management
        
    def store_document(self, doc_result: DocumentResult) -> str:
        # Save document with FIPS-compliant ID
        # Update indexes
        # Return document ID
        
    def load_document(self, doc_id: str) -> Optional[DocumentResult]:
        # Lazy load with memory management
        # Cache frequently accessed docs
        
    def search_documents(self, **filters) -> List[DocumentMetadata]:
        # Search by filename, date, size, etc.
        # Use indexes for efficiency
        
    def get_chunks_by_page(self, doc_id: str, page_num: int) -> List[ChunkData]:
        # Retrieve specific page chunks
        # Memory-efficient chunk loading
```

### Memory Management Strategy
```python
class MemoryManager:
    def __init__(self, max_memory_mb: int):
        self.max_memory = max_memory_mb * 1024 * 1024
        self.loaded_docs = OrderedDict()  # LRU cache
        self.memory_usage = 0
        
    def get_document(self, doc_id: str):
        # Load if not cached, evict LRU if needed
        
    def evict_oldest(self):
        # Remove oldest documents from memory
        
    def cleanup_memory(self):
        # Force memory cleanup when threshold reached
```

### Index Structure Examples
```python
# document_index.json
{
    "doc_abcd1234": {
        "filename": "report.pdf",
        "upload_date": "2024-01-01T00:00:00Z",
        "total_pages": 15,
        "total_chunks": 23,
        "file_size": 1024000,
        "status": "processed"
    }
}

# chunk_index.json
{
    "doc_abcd1234": {
        "chunks_by_page": {
            "1": ["chunk_001", "chunk_002"],
            "2": ["chunk_003", "chunk_004"]
        },
        "total_chunks": 23
    }
}
```

## Success Criteria for This Chat
- ✅ File-based storage system handles 75+ documents efficiently
- ✅ Memory usage stays under 100MB limit
- ✅ Document indexing enables fast lookup and filtering
- ✅ Storage utilities provide monitoring and management
- ✅ Integration with document processor is seamless
- ✅ All operations remain FIPS-compliant

## Memory Efficiency Requirements
- Load only needed documents into memory
- Implement LRU cache for frequently accessed content
- Stream large document processing
- Monitor and limit memory usage
- Graceful handling when memory limits reached

## Next Chat Guide
After completing the storage system, proceed to **Chat Guide 03: Search & Retrieval Engine** to implement similarity search over the stored documents.

## Notes for Implementation
- Design for 300 pages across 75 PDFs use case
- Prioritize memory efficiency over speed when needed
- Use JSON for all storage (human-readable, debuggable)
- Implement comprehensive error handling
- Monitor disk usage and provide cleanup tools
- Ensure all file operations are atomic where possible