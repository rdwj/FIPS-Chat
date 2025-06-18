"""
Comprehensive tests for the FileStorage system.
Tests memory management, indexing, storage operations, and integration.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from rag.file_storage import FileStorage, MemoryManager, StorageStats, DocumentIndex
from rag.document_processor import (
    DocumentResult, DocumentMetadata, DocumentChunk, ChunkMetadata, 
    DoclingProcessor, RAGConfig
)


class TestMemoryManager(unittest.TestCase):
    """Test the MemoryManager class."""
    
    def setUp(self):
        """Set up test environment."""
        self.memory_manager = MemoryManager(max_memory_mb=1)  # 1MB limit for testing
        
    def test_memory_manager_initialization(self):
        """Test memory manager initialization."""
        self.assertEqual(self.memory_manager.max_memory_bytes, 1024 * 1024)
        self.assertEqual(self.memory_manager.current_memory_usage, 0)
        self.assertEqual(len(self.memory_manager.loaded_docs), 0)
        
    def test_document_loading_and_caching(self):
        """Test document loading and LRU caching."""
        # Create mock documents
        doc1 = self._create_mock_document("doc1", "Small content")
        doc2 = self._create_mock_document("doc2", "Another small content")
        
        # Create loader functions
        def loader1(doc_id):
            return doc1 if doc_id == "doc1" else None
            
        def loader2(doc_id):
            return doc2 if doc_id == "doc2" else None
        
        # Load first document
        result1 = self.memory_manager.get_document("doc1", loader1)
        self.assertEqual(result1, doc1)
        self.assertEqual(len(self.memory_manager.loaded_docs), 1)
        
        # Load second document
        result2 = self.memory_manager.get_document("doc2", loader2)
        self.assertEqual(result2, doc2)
        self.assertEqual(len(self.memory_manager.loaded_docs), 2)
        
        # Access first document again (should be cached)
        result1_cached = self.memory_manager.get_document("doc1", loader1)
        self.assertEqual(result1_cached, doc1)
        
        # Check cache stats
        stats = self.memory_manager.get_cache_stats()
        self.assertEqual(stats['loaded_documents'], 2)
        self.assertGreater(stats['cache_hit_rate'], 0)
        
    def test_memory_eviction(self):
        """Test LRU eviction when memory limit reached."""
        # Create a large document that will exceed memory limit
        large_content = "X" * (500 * 1024)  # 500KB
        doc1 = self._create_mock_document("doc1", large_content)
        doc2 = self._create_mock_document("doc2", large_content)
        doc3 = self._create_mock_document("doc3", large_content)
        
        def loader(doc_id):
            if doc_id == "doc1":
                return doc1
            elif doc_id == "doc2":
                return doc2
            elif doc_id == "doc3":
                return doc3
            return None
        
        # Load documents - should trigger eviction
        self.memory_manager.get_document("doc1", loader)
        self.memory_manager.get_document("doc2", loader)
        self.memory_manager.get_document("doc3", loader)  # Should evict doc1
        
        # doc1 should be evicted, doc2 and doc3 should remain
        self.assertNotIn("doc1", self.memory_manager.loaded_docs)
        self.assertIn("doc2", self.memory_manager.loaded_docs)
        self.assertIn("doc3", self.memory_manager.loaded_docs)
        
    def test_manual_eviction(self):
        """Test manual document eviction."""
        doc1 = self._create_mock_document("doc1", "Content")
        
        def loader(doc_id):
            return doc1 if doc_id == "doc1" else None
        
        # Load document
        self.memory_manager.get_document("doc1", loader)
        self.assertIn("doc1", self.memory_manager.loaded_docs)
        
        # Manually evict
        self.memory_manager.evict_document("doc1")
        self.assertNotIn("doc1", self.memory_manager.loaded_docs)
        
    def test_clear_cache(self):
        """Test clearing all cached documents."""
        doc1 = self._create_mock_document("doc1", "Content1")
        doc2 = self._create_mock_document("doc2", "Content2")
        
        def loader(doc_id):
            return doc1 if doc_id == "doc1" else doc2
        
        # Load documents
        self.memory_manager.get_document("doc1", loader)
        self.memory_manager.get_document("doc2", loader)
        
        # Clear cache
        self.memory_manager.clear_cache()
        self.assertEqual(len(self.memory_manager.loaded_docs), 0)
        self.assertEqual(self.memory_manager.current_memory_usage, 0)
        
    def _create_mock_document(self, doc_id: str, content: str) -> DocumentResult:
        """Create a mock document for testing."""
        chunk_metadata = ChunkMetadata(
            chunk_id="chunk_001",
            page_number=1,
            section_title="Test Section",
            char_start=0,
            char_end=len(content),
            word_count=len(content.split())
        )
        
        chunk = DocumentChunk(text=content, metadata=chunk_metadata)
        
        doc_metadata = DocumentMetadata(
            doc_id=doc_id,
            filename="test.txt",
            upload_timestamp=datetime.now().isoformat(),
            total_pages=1,
            total_chunks=1,
            processing_status="completed",
            file_size_bytes=len(content.encode('utf-8'))
        )
        
        return DocumentResult(metadata=doc_metadata, chunks=[chunk])


class TestFileStorage(unittest.TestCase):
    """Test the FileStorage class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = FileStorage(storage_path=self.temp_dir, max_memory_mb=1)
        
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_storage_initialization(self):
        """Test storage system initialization."""
        self.assertTrue(self.storage.documents_path.exists())
        self.assertTrue(self.storage.indexes_path.exists())
        self.assertTrue(self.storage.config_path.exists())
        self.assertEqual(len(self.storage.document_index), 0)
        
    def test_document_storage_and_retrieval(self):
        """Test storing and retrieving documents."""
        # Create test document
        doc_result = self._create_test_document("test_doc", "Test content for storage")
        
        # Store document
        doc_id = self.storage.store_document(doc_result)
        self.assertEqual(doc_id, doc_result.metadata.doc_id)
        
        # Verify document is in index
        self.assertIn(doc_id, self.storage.document_index)
        
        # Retrieve document
        retrieved_doc = self.storage.load_document(doc_id)
        self.assertIsNotNone(retrieved_doc)
        if retrieved_doc:
            self.assertEqual(retrieved_doc.metadata.doc_id, doc_id)
            self.assertEqual(len(retrieved_doc.chunks), len(doc_result.chunks))
        
    def test_document_indexing(self):
        """Test document indexing functionality."""
        # Create and store multiple documents
        doc1 = self._create_test_document("doc1", "Python programming tutorial")
        doc2 = self._create_test_document("doc2", "JavaScript web development")
        doc3 = self._create_test_document("doc3", "Python data science guide")
        
        self.storage.store_document(doc1)
        self.storage.store_document(doc2)
        self.storage.store_document(doc3)
        
        # Test search functionality
        python_docs = self.storage.search_documents("Python")
        self.assertEqual(len(python_docs), 2)  # doc1 and doc3
        
        js_docs = self.storage.search_documents("JavaScript")
        self.assertEqual(len(js_docs), 1)  # doc2
        
    def test_document_listing_and_filtering(self):
        """Test document listing with filters."""
        # Create documents with different properties
        doc1 = self._create_test_document("doc1", "Content", filename="report.pdf", pages=10)
        doc2 = self._create_test_document("doc2", "Content", filename="guide.pdf", pages=5)
        doc3 = self._create_test_document("doc3", "Content", filename="manual.txt", pages=20)
        
        self.storage.store_document(doc1)
        self.storage.store_document(doc2)
        self.storage.store_document(doc3)
        
        # Test filtering by filename
        pdf_docs = self.storage.list_documents(filename_contains="pdf")
        self.assertEqual(len(pdf_docs), 2)
        
        # Test filtering by pages
        large_docs = self.storage.list_documents(min_pages=10)
        self.assertEqual(len(large_docs), 2)  # doc1 and doc3
        
        small_docs = self.storage.list_documents(max_pages=10)
        self.assertEqual(len(small_docs), 2)  # doc1 and doc2
        
    def test_document_deletion(self):
        """Test document deletion."""
        # Create and store document
        doc_result = self._create_test_document("test_doc", "Content to delete")
        doc_id = self.storage.store_document(doc_result)
        
        # Verify document exists
        self.assertIn(doc_id, self.storage.document_index)
        retrieved_doc = self.storage.load_document(doc_id)
        self.assertIsNotNone(retrieved_doc)
        
        # Delete document
        deleted = self.storage.delete_document(doc_id)
        self.assertTrue(deleted)
        
        # Verify document is gone
        self.assertNotIn(doc_id, self.storage.document_index)
        retrieved_doc = self.storage.load_document(doc_id)
        self.assertIsNone(retrieved_doc)
        
    def test_chunk_retrieval_by_page(self):
        """Test retrieving chunks by page number."""
        # Create document with multiple pages
        doc_result = self._create_multi_page_document()
        doc_id = self.storage.store_document(doc_result)
        
        # Get chunks for specific pages
        page1_chunks = self.storage.get_chunks_by_page(doc_id, 1)
        page2_chunks = self.storage.get_chunks_by_page(doc_id, 2)
        
        self.assertGreater(len(page1_chunks), 0)
        self.assertGreater(len(page2_chunks), 0)
        
        # Verify chunks are from correct pages
        for chunk in page1_chunks:
            self.assertEqual(chunk.metadata.page_number, 1)
        for chunk in page2_chunks:
            self.assertEqual(chunk.metadata.page_number, 2)
            
    def test_storage_statistics(self):
        """Test storage statistics calculation."""
        # Initially empty
        stats = self.storage.get_storage_stats()
        self.assertEqual(stats.total_documents, 0)
        self.assertEqual(stats.total_chunks, 0)
        
        # Add documents
        doc1 = self._create_test_document("doc1", "Content 1")
        doc2 = self._create_test_document("doc2", "Content 2")
        
        self.storage.store_document(doc1)
        self.storage.store_document(doc2)
        
        # Check updated stats
        stats = self.storage.get_storage_stats()
        self.assertEqual(stats.total_documents, 2)
        self.assertEqual(stats.total_chunks, 2)  # Each doc has 1 chunk
        self.assertGreater(stats.total_disk_usage_bytes, 0)
        
    def test_storage_cleanup(self):
        """Test storage cleanup functionality."""
        # Create old document
        old_doc = self._create_test_document("old_doc", "Old content")
        old_doc.metadata.upload_timestamp = (datetime.now() - timedelta(days=35)).isoformat()
        self.storage.store_document(old_doc)
        
        # Create recent document
        new_doc = self._create_test_document("new_doc", "New content")
        self.storage.store_document(new_doc)
        
        # Cleanup old documents (older than 30 days)
        cleanup_result = self.storage.cleanup_storage(max_age_days=30)
        
        # Check results
        self.assertEqual(cleanup_result['removed_old_documents'], 1)
        self.assertNotIn(old_doc.metadata.doc_id, self.storage.document_index)
        self.assertIn(new_doc.metadata.doc_id, self.storage.document_index)
        
    def test_storage_validation(self):
        """Test storage integrity validation."""
        # Add valid document
        doc_result = self._create_test_document("valid_doc", "Valid content")
        self.storage.store_document(doc_result)
        
        # Run validation
        validation_result = self.storage.validate_storage()
        
        self.assertEqual(validation_result['valid_documents'], 1)
        self.assertEqual(validation_result['corrupt_documents'], 0)
        self.assertEqual(validation_result['missing_files'], 0)
        self.assertEqual(len(validation_result['errors']), 0)
        
    def test_document_export(self):
        """Test document export functionality."""
        # Create and store documents
        doc1 = self._create_test_document("doc1", "Export content 1")
        doc2 = self._create_test_document("doc2", "Export content 2")
        
        self.storage.store_document(doc1)
        self.storage.store_document(doc2)
        
        # Export documents
        export_dir = os.path.join(self.temp_dir, "export")
        success = self.storage.export_documents(export_dir)
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(export_dir))
        
        # Check exported files
        export_files = list(Path(export_dir).glob("export_*.json"))
        self.assertEqual(len(export_files), 2)
        
        # Check indexes export
        indexes_file = Path(export_dir) / "indexes_export.json"
        self.assertTrue(indexes_file.exists())
        
    def test_index_persistence(self):
        """Test that indexes are properly saved and loaded."""
        # Create and store document
        doc_result = self._create_test_document("test_doc", "Index persistence test")
        doc_id = self.storage.store_document(doc_result)
        
        # Create new storage instance (should load existing indexes)
        new_storage = FileStorage(storage_path=self.temp_dir, max_memory_mb=1)
        
        # Verify indexes were loaded
        self.assertIn(doc_id, new_storage.document_index)
        self.assertIn(doc_id, new_storage.chunk_index)
        
        # Verify document can be retrieved
        retrieved_doc = new_storage.load_document(doc_id)
        self.assertIsNotNone(retrieved_doc)
        self.assertEqual(retrieved_doc.metadata.doc_id, doc_id)
        
    def _create_test_document(self, doc_id: str, content: str, 
                            filename: str = "test.txt", pages: int = 1) -> DocumentResult:
        """Create a test document."""
        chunk_metadata = ChunkMetadata(
            chunk_id="chunk_001",
            page_number=1,
            section_title="Test Section",
            char_start=0,
            char_end=len(content),
            word_count=len(content.split())
        )
        
        chunk = DocumentChunk(text=content, metadata=chunk_metadata)
        
        doc_metadata = DocumentMetadata(
            doc_id=doc_id,
            filename=filename,
            upload_timestamp=datetime.now().isoformat(),
            total_pages=pages,
            total_chunks=1,
            processing_status="completed",
            file_size_bytes=len(content.encode('utf-8'))
        )
        
        return DocumentResult(metadata=doc_metadata, chunks=[chunk])
        
    def _create_multi_page_document(self) -> DocumentResult:
        """Create a document with multiple pages."""
        chunks = []
        
        # Create chunks for different pages
        for page in range(1, 3):  # Pages 1 and 2
            for chunk_num in range(1, 3):  # 2 chunks per page
                chunk_id = f"chunk_{page}_{chunk_num:02d}"
                content = f"Page {page} content chunk {chunk_num}"
                
                chunk_metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    page_number=page,
                    section_title=f"Section {page}",
                    char_start=(page-1)*100 + (chunk_num-1)*50,
                    char_end=(page-1)*100 + chunk_num*50,
                    word_count=len(content.split())
                )
                
                chunk = DocumentChunk(text=content, metadata=chunk_metadata)
                chunks.append(chunk)
        
        doc_metadata = DocumentMetadata(
            doc_id="multi_page_doc",
            filename="multipage.pdf",
            upload_timestamp=datetime.now().isoformat(),
            total_pages=2,
            total_chunks=len(chunks),
            processing_status="completed",
            file_size_bytes=1000
        )
        
        return DocumentResult(metadata=doc_metadata, chunks=chunks)


class TestStorageIntegration(unittest.TestCase):
    """Test integration between FileStorage and DoclingProcessor."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = FileStorage(storage_path=self.temp_dir, max_memory_mb=1)
        
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    @patch('rag.document_processor.DOCLING_AVAILABLE', True)
    def test_processor_with_storage_integration(self):
        """Test DoclingProcessor with storage integration."""
        # Mock the Docling converter
        with patch('rag.document_processor.DocumentConverter') as mock_converter:
            # Create processor with storage
            processor = DoclingProcessor(storage=self.storage, auto_store=True)
            
            # Test that storage is properly set
            self.assertEqual(processor.storage, self.storage)
            self.assertTrue(processor.auto_store)
            
    def test_process_and_store_text(self):
        """Test processing and storing text directly."""
        with patch('rag.document_processor.DOCLING_AVAILABLE', True):
            with patch('rag.document_processor.DocumentConverter'):
                processor = DoclingProcessor(storage=self.storage)
                
                # Process and store text
                doc_id = processor.process_and_store_text("Test content", "test.txt")
                
                # Verify document was stored
                self.assertIn(doc_id, self.storage.document_index)
                retrieved_doc = self.storage.load_document(doc_id)
                self.assertIsNotNone(retrieved_doc)
                
    def test_document_deduplication(self):
        """Test document deduplication based on content."""
        with patch('rag.document_processor.DOCLING_AVAILABLE', True):
            with patch('rag.document_processor.DocumentConverter'):
                processor = DoclingProcessor(storage=self.storage)
                
                content = "Duplicate content test"
                
                # Process same content twice
                doc_id1 = processor.process_and_store_text(content, "file1.txt")
                doc_id2 = processor.process_and_store_text(content, "file2.txt")
                
                # Should have same document ID (content-based)
                self.assertEqual(doc_id1, doc_id2)
                
                # Should only have one document in storage
                self.assertEqual(len(self.storage.document_index), 1)
                
    def test_check_document_exists(self):
        """Test checking if document exists."""
        with patch('rag.document_processor.DOCLING_AVAILABLE', True):
            with patch('rag.document_processor.DocumentConverter'):
                processor = DoclingProcessor(storage=self.storage)
                
                content = "Existing document content"
                
                # Check before storing
                exists_before = processor.check_document_exists(content)
                self.assertIsNone(exists_before)
                
                # Store document
                doc_id = processor.process_and_store_text(content, "test.txt")
                
                # Check after storing
                exists_after = processor.check_document_exists(content)
                self.assertEqual(exists_after, doc_id)


if __name__ == '__main__':
    unittest.main() 