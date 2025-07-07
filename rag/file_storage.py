"""
File-based storage system for RAG documents and chunks.
Provides memory-efficient document storage with FIPS-compliant operations.
"""

import json
import logging
import os
import shutil
import time
from collections import OrderedDict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import psutil
import threading
import hashlib

from .document_processor import DocumentResult, DocumentMetadata, DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class StorageStats:
    """Statistics about storage usage."""
    total_documents: int
    total_chunks: int
    total_disk_usage_bytes: int
    memory_usage_bytes: int
    cache_hit_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class DocumentIndex:
    """Index entry for a document."""
    doc_id: str
    filename: str
    upload_date: str
    total_pages: int
    total_chunks: int
    file_size_bytes: int
    status: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MemoryManager:
    """Manages memory usage for loaded documents."""
    
    def __init__(self, max_memory_mb: int = 100):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.loaded_docs: OrderedDict[str, DocumentResult] = OrderedDict()
        self.doc_sizes: Dict[str, int] = {}
        self.current_memory_usage = 0
        self.access_count = 0
        self.cache_hits = 0
        self.lock = threading.RLock()
        
    def get_document(self, doc_id: str, loader_func) -> Optional[DocumentResult]:
        """Get document from cache or load it."""
        with self.lock:
            self.access_count += 1
            
            if doc_id in self.loaded_docs:
                # Move to end (most recently used)
                self.loaded_docs.move_to_end(doc_id)
                self.cache_hits += 1
                return self.loaded_docs[doc_id]
            
            # Load document
            doc_result = loader_func(doc_id)
            if doc_result is None:
                return None
                
            # Calculate memory usage
            doc_size = self._estimate_document_size(doc_result)
            
            # Ensure we have space
            self._ensure_memory_available(doc_size)
            
            # Add to cache
            self.loaded_docs[doc_id] = doc_result
            self.doc_sizes[doc_id] = doc_size
            self.current_memory_usage += doc_size
            
            return doc_result
    
    def _estimate_document_size(self, doc_result: DocumentResult) -> int:
        """Estimate memory usage of a document."""
        try:
            # Simple estimation based on JSON size
            json_str = json.dumps(doc_result.to_dict())
            return len(json_str.encode('utf-8'))
        except Exception:
            # Fallback estimation
            total_text = sum(len(chunk.text) for chunk in doc_result.chunks)
            return total_text * 2  # Rough multiplier for overhead
    
    def _ensure_memory_available(self, needed_bytes: int):
        """Ensure enough memory is available by evicting LRU documents."""
        with self.lock:
            while (self.current_memory_usage + needed_bytes > self.max_memory_bytes 
                   and self.loaded_docs):
                # Remove oldest document
                doc_id, _ = self.loaded_docs.popitem(last=False)
                freed_bytes = self.doc_sizes.pop(doc_id, 0)
                self.current_memory_usage -= freed_bytes
                logger.debug(f"Evicted document {doc_id}, freed {freed_bytes} bytes")
    
    def evict_document(self, doc_id: str):
        """Manually evict a specific document."""
        with self.lock:
            if doc_id in self.loaded_docs:
                del self.loaded_docs[doc_id]
                freed_bytes = self.doc_sizes.pop(doc_id, 0)
                self.current_memory_usage -= freed_bytes
    
    def clear_cache(self):
        """Clear all cached documents."""
        with self.lock:
            self.loaded_docs.clear()
            self.doc_sizes.clear()
            self.current_memory_usage = 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            hit_rate = self.cache_hits / max(self.access_count, 1)
            return {
                'memory_usage_bytes': self.current_memory_usage,
                'loaded_documents': len(self.loaded_docs),
                'cache_hit_rate': hit_rate,
                'max_memory_bytes': self.max_memory_bytes
            }


class FileStorage:
    """File-based storage system for RAG documents."""
    
    def __init__(self, storage_path: str = "rag_storage", max_memory_mb: int = 100):
        """Initialize file storage system."""
        self.storage_path = Path(storage_path)
        self.documents_path = self.storage_path / "documents"
        self.indexes_path = self.storage_path / "indexes"
        self.config_path = self.storage_path / "config"
        
        # Create directories if they don't exist
        self._ensure_directories()
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(max_memory_mb)
        
        # Initialize indexes
        self.document_index: Dict[str, DocumentIndex] = {}
        self.chunk_index: Dict[str, Dict] = {}
        self.term_index: Dict[str, List[str]] = {}
        
        # Load existing indexes
        self._load_indexes()
        
        logger.info(f"FileStorage initialized at {storage_path}")
    
    def _ensure_directories(self):
        """Ensure all required directories exist."""
        for path in [self.documents_path, self.indexes_path, self.config_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def _generate_doc_filename(self, doc_id: str) -> str:
        """Generate filename for document storage."""
        return f"doc_{doc_id[:8]}.json"
    
    def _load_indexes(self):
        """Load indexes from disk."""
        try:
            # Load document index
            doc_index_path = self.indexes_path / "document_index.json"
            if doc_index_path.exists():
                with open(doc_index_path, 'r') as f:
                    data = json.load(f)
                    self.document_index = {
                        doc_id: DocumentIndex(**info) 
                        for doc_id, info in data.items()
                    }
            
            # Load chunk index
            chunk_index_path = self.indexes_path / "chunk_index.json"
            if chunk_index_path.exists():
                with open(chunk_index_path, 'r') as f:
                    self.chunk_index = json.load(f)
            
            # Load term index
            term_index_path = self.indexes_path / "term_index.json"
            if term_index_path.exists():
                with open(term_index_path, 'r') as f:
                    self.term_index = json.load(f)
                    
            logger.info(f"Loaded indexes: {len(self.document_index)} documents")
            
        except Exception as e:
            logger.error(f"Error loading indexes: {e}")
            # Initialize empty indexes on error
            self.document_index = {}
            self.chunk_index = {}
            self.term_index = {}
    
    def _save_indexes(self):
        """Save indexes to disk."""
        try:
            # Save document index
            doc_index_path = self.indexes_path / "document_index.json"
            with open(doc_index_path, 'w') as f:
                json.dump(
                    {doc_id: idx.to_dict() for doc_id, idx in self.document_index.items()},
                    f, indent=2
                )
            
            # Save chunk index
            chunk_index_path = self.indexes_path / "chunk_index.json"
            with open(chunk_index_path, 'w') as f:
                json.dump(self.chunk_index, f, indent=2)
            
            # Save term index
            term_index_path = self.indexes_path / "term_index.json"
            with open(term_index_path, 'w') as f:
                json.dump(self.term_index, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving indexes: {e}")
            raise
    
    def store_document(self, doc_result: DocumentResult) -> str:
        """Store a document and return its ID."""
        try:
            doc_id = doc_result.metadata.doc_id
            filename = self._generate_doc_filename(doc_id)
            doc_path = self.documents_path / filename
            
            # Check if document already exists
            if doc_path.exists():
                logger.warning(f"Document {doc_id} already exists, overwriting")
            
            # Save document to disk
            with open(doc_path, 'w') as f:
                json.dump(doc_result.to_dict(), f, indent=2)
            
            # Update document index
            self.document_index[doc_id] = DocumentIndex(
                doc_id=doc_id,
                filename=doc_result.metadata.filename,
                upload_date=doc_result.metadata.upload_timestamp,
                total_pages=doc_result.metadata.total_pages,
                total_chunks=doc_result.metadata.total_chunks,
                file_size_bytes=doc_result.metadata.file_size_bytes,
                status=doc_result.metadata.processing_status
            )
            
            # Update chunk index
            chunks_by_page = {}
            for chunk in doc_result.chunks:
                page_num = str(chunk.metadata.page_number)
                if page_num not in chunks_by_page:
                    chunks_by_page[page_num] = []
                chunks_by_page[page_num].append(chunk.metadata.chunk_id)
            
            self.chunk_index[doc_id] = {
                "chunks_by_page": chunks_by_page,
                "total_chunks": len(doc_result.chunks)
            }
            
            # Update term index (basic implementation)
            self._update_term_index(doc_id, doc_result)
            
            # Save indexes
            self._save_indexes()
            
            # Clear from memory cache to force reload with new data
            self.memory_manager.evict_document(doc_id)
            
            logger.info(f"Stored document {doc_id} with {len(doc_result.chunks)} chunks")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error storing document: {e}")
            raise
    
    def _update_term_index(self, doc_id: str, doc_result: DocumentResult):
        """Update term index with document terms."""
        try:
            # Simple term extraction - can be enhanced
            all_text = " ".join(chunk.text for chunk in doc_result.chunks)
            # Basic tokenization
            terms = set(word.lower().strip('.,!?;:"()[]{}') 
                       for word in all_text.split() 
                       if len(word) > 2)
            
            for term in terms:
                if term not in self.term_index:
                    self.term_index[term] = []
                if doc_id not in self.term_index[term]:
                    self.term_index[term].append(doc_id)
                    
        except Exception as e:
            logger.error(f"Error updating term index: {e}")
    
    def load_document(self, doc_id: str) -> Optional[DocumentResult]:
        """Load a document with memory management."""
        def _load_from_disk(doc_id: str) -> Optional[DocumentResult]:
            try:
                filename = self._generate_doc_filename(doc_id)
                doc_path = self.documents_path / filename
                
                if not doc_path.exists():
                    return None
                
                with open(doc_path, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct DocumentResult from JSON data
                from .document_processor import DocumentMetadata, DocumentChunk, ChunkMetadata
                
                metadata = DocumentMetadata(**data['metadata'])
                chunks = []
                
                for chunk_data in data['chunks']:
                    chunk_metadata = ChunkMetadata(**chunk_data['metadata'])
                    chunk = DocumentChunk(
                        text=chunk_data['text'],
                        metadata=chunk_metadata
                    )
                    chunks.append(chunk)
                
                return DocumentResult(metadata=metadata, chunks=chunks)
                
            except Exception as e:
                logger.error(f"Error loading document {doc_id}: {e}")
                return None
        
        return self.memory_manager.get_document(doc_id, _load_from_disk)
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its indexes."""
        try:
            filename = self._generate_doc_filename(doc_id)
            doc_path = self.documents_path / filename
            
            if not doc_path.exists():
                logger.warning(f"Document {doc_id} not found for deletion")
                return False
            
            # Remove file
            doc_path.unlink()
            
            # Remove from indexes
            self.document_index.pop(doc_id, None)
            self.chunk_index.pop(doc_id, None)
            
            # Remove from term index
            for term, doc_ids in self.term_index.items():
                if doc_id in doc_ids:
                    doc_ids.remove(doc_id)
            
            # Clean up empty term entries
            self.term_index = {
                term: doc_ids for term, doc_ids in self.term_index.items()
                if doc_ids
            }
            
            # Remove from memory cache
            self.memory_manager.evict_document(doc_id)
            
            # Save updated indexes
            self._save_indexes()
            
            logger.info(f"Deleted document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False
    
    def list_documents(self, **filters) -> List[DocumentIndex]:
        """List documents with optional filtering."""
        documents = list(self.document_index.values())
        
        # Apply filters
        if 'filename_contains' in filters:
            term = filters['filename_contains'].lower()
            documents = [d for d in documents if term in d.filename.lower()]
        
        if 'status' in filters:
            status = filters['status']
            documents = [d for d in documents if d.status == status]
        
        if 'min_pages' in filters:
            min_pages = filters['min_pages']
            documents = [d for d in documents if d.total_pages >= min_pages]
        
        if 'max_pages' in filters:
            max_pages = filters['max_pages']
            documents = [d for d in documents if d.total_pages <= max_pages]
        
        # Sort by upload date (newest first)
        documents.sort(key=lambda x: x.upload_date, reverse=True)
        
        return documents
    
    def search_documents(self, query: str) -> List[DocumentIndex]:
        """Search documents by term."""
        query_terms = set(query.lower().split())
        matching_doc_ids = set()
        
        for term in query_terms:
            if term in self.term_index:
                matching_doc_ids.update(self.term_index[term])
        
        # Return document indexes for matching documents
        return [
            self.document_index[doc_id] 
            for doc_id in matching_doc_ids 
            if doc_id in self.document_index
        ]
    
    def get_chunks_by_page(self, doc_id: str, page_num: int) -> List[DocumentChunk]:
        """Get chunks for a specific page."""
        doc_result = self.load_document(doc_id)
        if not doc_result:
            return []
        
        return [
            chunk for chunk in doc_result.chunks 
            if chunk.metadata.page_number == page_num
        ]
    
    def get_storage_stats(self) -> StorageStats:
        """Get storage statistics."""
        try:
            # Calculate disk usage
            total_size = 0
            for file_path in self.storage_path.rglob('*.json'):
                total_size += file_path.stat().st_size
            
            # Get memory stats
            cache_stats = self.memory_manager.get_cache_stats()
            
            # Calculate totals
            total_chunks = sum(
                doc_index.total_chunks 
                for doc_index in self.document_index.values()
            )
            
            return StorageStats(
                total_documents=len(self.document_index),
                total_chunks=total_chunks,
                total_disk_usage_bytes=total_size,
                memory_usage_bytes=cache_stats['memory_usage_bytes'],
                cache_hit_rate=cache_stats['cache_hit_rate']
            )
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return StorageStats(0, 0, 0, 0, 0.0)
    
    def cleanup_storage(self, max_age_days: int = 30) -> Dict[str, int]:
        """Clean up old or orphaned files."""
        try:
            current_time = datetime.now()
            removed_files = 0
            removed_docs = 0
            
            # Find orphaned document files
            existing_files = set(self.documents_path.glob('doc_*.json'))
            expected_files = {
                self.documents_path / self._generate_doc_filename(doc_id)
                for doc_id in self.document_index.keys()
            }
            
            orphaned_files = existing_files - expected_files
            
            # Remove orphaned files
            for file_path in orphaned_files:
                try:
                    file_path.unlink()
                    removed_files += 1
                    logger.info(f"Removed orphaned file: {file_path.name}")
                except Exception as e:
                    logger.error(f"Error removing orphaned file {file_path}: {e}")
            
            # Find old documents (if max_age_days specified)
            if max_age_days > 0:
                cutoff_time = current_time.timestamp() - (max_age_days * 24 * 3600)
                old_doc_ids = []
                
                for doc_id, doc_index in self.document_index.items():
                    try:
                        upload_time = datetime.fromisoformat(doc_index.upload_date).timestamp()
                        if upload_time < cutoff_time:
                            old_doc_ids.append(doc_id)
                    except Exception:
                        # Skip documents with invalid dates
                        continue
                
                # Remove old documents
                for doc_id in old_doc_ids:
                    if self.delete_document(doc_id):
                        removed_docs += 1
            
            logger.info(f"Cleanup completed: {removed_files} orphaned files, {removed_docs} old documents removed")
            
            return {
                'removed_orphaned_files': removed_files,
                'removed_old_documents': removed_docs
            }
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return {'removed_orphaned_files': 0, 'removed_old_documents': 0}
    
    def validate_storage(self) -> Dict[str, Any]:
        """Validate storage integrity."""
        try:
            results = {
                'valid_documents': 0,
                'corrupt_documents': 0,
                'missing_files': 0,
                'index_mismatches': 0,
                'errors': []
            }
            
            # Check each document in the index
            for doc_id, doc_index in self.document_index.items():
                try:
                    # Check if file exists
                    filename = self._generate_doc_filename(doc_id)
                    doc_path = self.documents_path / filename
                    
                    if not doc_path.exists():
                        results['missing_files'] += 1
                        results['errors'].append(f"Missing file for document {doc_id}")
                        continue
                    
                    # Try to load document
                    doc_result = self.load_document(doc_id)
                    if not doc_result:
                        results['corrupt_documents'] += 1
                        results['errors'].append(f"Cannot load document {doc_id}")
                        continue
                    
                    # Validate index matches document
                    if doc_result.metadata.total_chunks != doc_index.total_chunks:
                        results['index_mismatches'] += 1
                        results['errors'].append(f"Chunk count mismatch for document {doc_id}")
                    
                    results['valid_documents'] += 1
                    
                except Exception as e:
                    results['corrupt_documents'] += 1
                    results['errors'].append(f"Error validating document {doc_id}: {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during storage validation: {e}")
            return {
                'valid_documents': 0,
                'corrupt_documents': 0,
                'missing_files': 0,
                'index_mismatches': 0,
                'errors': [f"Validation failed: {str(e)}"]
            }
    
    def export_documents(self, output_path: str, doc_ids: Optional[List[str]] = None) -> bool:
        """Export documents to a backup location."""
        try:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine which documents to export
            if doc_ids is None:
                doc_ids = list(self.document_index.keys())
            
            exported_count = 0
            
            for doc_id in doc_ids:
                try:
                    doc_result = self.load_document(doc_id)
                    if doc_result:
                        # Export document
                        export_filename = f"export_{self._generate_doc_filename(doc_id)}"
                        export_path = output_dir / export_filename
                        
                        with open(export_path, 'w') as f:
                            json.dump(doc_result.to_dict(), f, indent=2)
                        
                        exported_count += 1
                        
                except Exception as e:
                    logger.error(f"Error exporting document {doc_id}: {e}")
            
            # Export indexes
            indexes_export = {
                'document_index': {doc_id: idx.to_dict() for doc_id, idx in self.document_index.items()},
                'chunk_index': self.chunk_index,
                'term_index': self.term_index
            }
            
            with open(output_dir / 'indexes_export.json', 'w') as f:
                json.dump(indexes_export, f, indent=2)
            
            logger.info(f"Exported {exported_count} documents to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error during export: {e}")
            return False 