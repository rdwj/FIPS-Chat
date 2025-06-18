"""
RAG (Retrieval-Augmented Generation) module for FIPS-Chat.

This module provides document processing, storage, and search capabilities
while maintaining FIPS compliance throughout all operations.
"""

from .document_processor import (
    DoclingProcessor,
    DocumentResult,
    DocumentMetadata,
    DocumentChunk,
    ChunkMetadata,
    RAGConfig
)

from .file_storage import (
    FileStorage,
    StorageStats,
    DocumentIndex,
    MemoryManager
)

from .search_engine import (
    TFIDFSearchEngine,
    SearchResult,
    SearchStats,
    SearchCache
)

__all__ = [
    # Document processing
    'DoclingProcessor',
    'DocumentResult',
    'DocumentMetadata', 
    'DocumentChunk',
    'ChunkMetadata',
    'RAGConfig',
    
    # File storage
    'FileStorage',
    'StorageStats',
    'DocumentIndex',
    'MemoryManager',
    
    # Search engine
    'TFIDFSearchEngine',
    'SearchResult',
    'SearchStats',
    'SearchCache'
] 