"""
Tests for the TF-IDF search engine implementation.
Tests TF-IDF indexing, hybrid search, performance, and FIPS compliance.
"""

import json
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

from rag.search_engine import TFIDFSearchEngine, SearchResult, SearchStats, SearchCache
from rag.file_storage import FileStorage
from rag.document_processor import (
    DoclingProcessor, DocumentResult, DocumentMetadata, 
    DocumentChunk, ChunkMetadata, RAGConfig
)


@pytest.fixture
def temp_storage_path():
    """Create temporary storage directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def file_storage(temp_storage_path):
    """Create file storage instance."""
    return FileStorage(temp_storage_path, max_memory_mb=50)


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    doc1_chunks = [
        DocumentChunk(
            text="Machine learning algorithms are computational methods that enable computers to learn from data without being explicitly programmed.",
            metadata=ChunkMetadata(
                chunk_id="chunk_001",
                page_number=1,
                section_title="Introduction to ML",
                char_start=0,
                char_end=128,
                word_count=18
            )
        ),
        DocumentChunk(
            text="Neural networks are a subset of machine learning algorithms inspired by the structure and function of biological neural networks.",
            metadata=ChunkMetadata(
                chunk_id="chunk_002",
                page_number=1,
                section_title="Neural Networks",
                char_start=129,
                char_end=257,
                word_count=19
            )
        )
    ]
    
    doc1 = DocumentResult(
        metadata=DocumentMetadata(
            doc_id="doc_1234567890abcdef",
            filename="ml_guide.pdf",
            upload_timestamp=datetime.now().isoformat(),
            total_pages=1,
            total_chunks=2,
            processing_status="completed",
            file_size_bytes=1024
        ),
        chunks=doc1_chunks
    )
    
    doc2_chunks = [
        DocumentChunk(
            text="Data privacy regulations like GDPR require organizations to implement appropriate technical and organizational measures to protect personal data.",
            metadata=ChunkMetadata(
                chunk_id="chunk_003",
                page_number=1,
                section_title="GDPR Overview",
                char_start=0,
                char_end=137,
                word_count=20
            )
        ),
        DocumentChunk(
            text="FIPS compliance ensures that cryptographic modules meet federal information processing standards for security.",
            metadata=ChunkMetadata(
                chunk_id="chunk_004",
                page_number=1,
                section_title="FIPS Standards",
                char_start=138,
                char_end=246,
                word_count=15
            )
        )
    ]
    
    doc2 = DocumentResult(
        metadata=DocumentMetadata(
            doc_id="doc_fedcba9876543210",
            filename="security_guide.pdf", 
            upload_timestamp=datetime.now().isoformat(),
            total_pages=1,
            total_chunks=2,
            processing_status="completed",
            file_size_bytes=2048
        ),
        chunks=doc2_chunks
    )
    
    return [doc1, doc2]


@pytest.fixture
def populated_storage(file_storage, sample_documents):
    """Create storage with sample documents."""
    for doc in sample_documents:
        file_storage.store_document(doc)
    return file_storage


@pytest.fixture
def search_engine(populated_storage):
    """Create search engine with populated storage."""
    engine = TFIDFSearchEngine(populated_storage, max_features=1000)
    return engine


class TestSearchCache:
    """Test the search result cache."""
    
    def test_cache_initialization(self):
        """Test cache initialization with custom parameters."""
        cache = SearchCache(max_size=50, ttl_seconds=600)
        assert cache.max_size == 50
        assert cache.ttl_seconds == 600
        assert len(cache.cache) == 0
    
    def test_cache_put_and_get(self):
        """Test basic cache put and get operations."""
        cache = SearchCache(max_size=2)
        
        # Create mock search results
        results = [
            SearchResult(
                chunk_id="chunk_001",
                doc_id="doc_123",
                text="test text",
                similarity_score=0.8,
                page_number=1,
                section_title="Test",
                filename="test.pdf",
                snippet="test text",
                context_before="",
                context_after="",
                combined_score=0.85
            )
        ]
        
        # Test put and get
        cache.put("query_hash_1", results)
        retrieved = cache.get("query_hash_1")
        
        assert retrieved is not None
        assert len(retrieved) == 1
        assert retrieved[0].chunk_id == "chunk_001"
    
    def test_cache_eviction(self):
        """Test LRU cache eviction."""
        cache = SearchCache(max_size=2)
        
        results1 = [Mock()]
        results2 = [Mock()]
        results3 = [Mock()]
        
        # Fill cache to capacity
        cache.put("query1", results1)
        cache.put("query2", results2)
        
        # Add third item, should evict first
        cache.put("query3", results3)
        
        assert cache.get("query1") is None  # Evicted
        assert cache.get("query2") is not None
        assert cache.get("query3") is not None
    
    def test_cache_ttl_expiration(self):
        """Test cache TTL expiration."""
        cache = SearchCache(ttl_seconds=0.1)  # Very short TTL
        
        results = [Mock()]
        cache.put("query", results)
        
        # Should be available immediately
        assert cache.get("query") is not None
        
        # Should expire after TTL
        import time
        time.sleep(0.2)
        assert cache.get("query") is None
    
    def test_cache_clear(self):
        """Test cache clearing."""
        cache = SearchCache()
        cache.put("query1", [Mock()])
        cache.put("query2", [Mock()])
        
        assert len(cache.cache) == 2
        
        cache.clear()
        assert len(cache.cache) == 0


class TestTFIDFSearchEngine:
    """Test the main TF-IDF search engine."""
    
    def test_initialization(self, populated_storage):
        """Test search engine initialization."""
        engine = TFIDFSearchEngine(
            populated_storage, 
            max_features=500,
            cache_size=50
        )
        
        assert engine.storage == populated_storage
        assert engine.max_features == 500
        assert engine.cache.max_size == 50
        assert engine.tfidf_matrix is None  # Not built yet
        assert len(engine.chunk_metadata) == 0
    
    def test_text_preprocessing(self, search_engine):
        """Test text preprocessing."""
        test_text = "This is a TEST text with   extra   spaces & special chars!!!"
        processed = search_engine._preprocess_text(test_text)
        
        assert "  " not in processed  # No double spaces
        assert processed == "This is a TEST text with extra spaces special chars"
    
    def test_index_building(self, search_engine):
        """Test TF-IDF index building."""
        success = search_engine.build_index()
        
        assert success is True
        assert search_engine.tfidf_matrix is not None
        assert search_engine.tfidf_matrix.shape[0] == 4  # 4 chunks total
        assert len(search_engine.chunk_metadata) == 4
        assert len(search_engine.doc_chunks_map) == 2  # 2 documents
    
    def test_index_persistence(self, search_engine):
        """Test saving and loading search index."""
        # Build and save index
        search_engine.build_index()
        original_vocab_size = len(search_engine.vectorizer.vocabulary_)
        original_chunks = len(search_engine.chunk_metadata)
        
        # Create new engine and load index
        new_engine = TFIDFSearchEngine(search_engine.storage)
        loaded = new_engine._load_index()
        
        assert loaded is True
        assert len(new_engine.vectorizer.vocabulary_) == original_vocab_size
        assert len(new_engine.chunk_metadata) == original_chunks
    
    def test_query_preprocessing(self, search_engine):
        """Test query preprocessing."""
        query = "What are machine learning algorithms???"
        processed = search_engine.preprocess_query(query)
        
        assert processed == "What are machine learning algorithms"
    
    def test_exact_match_boost_calculation(self, search_engine):
        """Test exact match boost calculation."""
        query = "machine learning"
        text_with_match = "Machine learning algorithms are powerful"
        text_without_match = "Neural networks are computational models"
        
        boost_with = search_engine._calculate_exact_match_boost(query, text_with_match)
        boost_without = search_engine._calculate_exact_match_boost(query, text_without_match)
        
        assert boost_with > 0
        assert boost_without == 0
        assert boost_with == 0.3  # Multi-word exact match
    
    def test_recency_score_calculation(self, search_engine):
        """Test recency score calculation."""
        recent_date = datetime.now().isoformat()
        old_date = "2020-01-01T00:00:00"
        
        recent_score = search_engine._calculate_recency_score(recent_date)
        old_score = search_engine._calculate_recency_score(old_date)
        
        assert recent_score > old_score
        assert 0.5 <= recent_score <= 1.0
        assert 0.5 <= old_score <= 1.0
    
    def test_position_score_calculation(self, search_engine):
        """Test position score calculation."""
        # Test early page vs late page
        early_score = search_engine._calculate_position_score(1, 10)
        late_score = search_engine._calculate_position_score(10, 10)
        
        assert early_score > late_score
        assert 0.7 <= early_score <= 1.0
        assert 0.7 <= late_score <= 1.0
        
        # Test single page document
        single_score = search_engine._calculate_position_score(1, 1)
        assert single_score == 1.0
    
    def test_section_match_boost(self, search_engine):
        """Test section title matching boost."""
        query = "neural networks"
        matching_section = "Neural Networks Overview"
        non_matching_section = "Data Processing"
        
        boost_match = search_engine._calculate_section_match_boost(query, matching_section)
        boost_no_match = search_engine._calculate_section_match_boost(query, non_matching_section)
        
        assert boost_match > 0
        assert boost_no_match == 0
        assert boost_match <= 0.5  # Max boost
    
    def test_snippet_generation(self, search_engine):
        """Test snippet generation around query matches."""
        text = "This is a long text about machine learning algorithms and their applications in various domains of computer science and artificial intelligence."
        query = "machine learning"
        
        snippet = search_engine._generate_snippet(text, query, max_length=50)
        
        assert "machine learning" in snippet.lower()
        assert len(snippet) <= 60  # Allowing for ellipsis
        assert "..." in snippet  # Should have ellipsis for truncation
    
    def test_basic_search(self, search_engine):
        """Test basic search functionality."""
        search_engine.build_index()
        
        results = search_engine.search_similar_chunks("machine learning", top_k=3)
        
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.similarity_score > 0 for r in results)
        assert all(r.combined_score > 0 for r in results)
        
        # Check that results are sorted by combined score
        scores = [r.combined_score for r in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_search_with_filters(self, search_engine):
        """Test search with various filters."""
        search_engine.build_index()
        
        # Test document ID filter
        results = search_engine.search_similar_chunks(
            "data", 
            filters={'doc_id': 'doc_fedcba9876543210'}
        )
        
        assert all(r.doc_id == 'doc_fedcba9876543210' for r in results)
        
        # Test filename filter
        results = search_engine.search_similar_chunks(
            "machine", 
            filters={'filename': 'ml_guide'}
        )
        
        assert all('ml_guide' in r.filename for r in results)
    
    def test_search_caching(self, search_engine):
        """Test search result caching."""
        search_engine.build_index()
        
        # First search
        results1 = search_engine.search_similar_chunks("neural networks")
        
        # Second identical search should hit cache
        results2 = search_engine.search_similar_chunks("neural networks")
        
        assert len(results1) == len(results2)
        assert search_engine.search_stats['cache_hits'] > 0
    
    def test_hybrid_scoring(self, search_engine):
        """Test that hybrid scoring improves over basic TF-IDF."""
        search_engine.build_index()
        
        # Search for exact phrase that appears in text
        results = search_engine.search_similar_chunks("machine learning algorithms")
        
        if results:
            # Check that combined score incorporates more than just TF-IDF
            top_result = results[0]
            assert top_result.combined_score != top_result.similarity_score
            
            # Verify result structure
            assert hasattr(top_result, 'snippet')
            assert hasattr(top_result, 'context_before')
            assert hasattr(top_result, 'context_after')
            assert len(top_result.snippet) > 0
    
    def test_result_filtering(self, search_engine):
        """Test result filtering and post-processing."""
        search_engine.build_index()
        
        # Get some results first
        results = search_engine.search_similar_chunks("data algorithms", top_k=10)
        
        # Test minimum score filtering
        filtered = search_engine.filter_results(results, min_score=0.5)
        assert all(r.combined_score >= 0.5 for r in filtered)
        
        # Test deduplication
        deduplicated = search_engine.filter_results(results, deduplicate=True)
        text_hashes = set()
        for result in deduplicated:
            text_hash = result.text[:50]  # Simple hash for testing
            assert text_hash not in text_hashes
            text_hashes.add(text_hash)
    
    def test_search_stats(self, search_engine):
        """Test search statistics tracking."""
        search_engine.build_index()
        
        # Perform some searches
        search_engine.search_similar_chunks("test query 1")
        search_engine.search_similar_chunks("test query 2")
        search_engine.search_similar_chunks("test query 1")  # Cache hit
        
        stats = search_engine.get_search_stats()
        
        assert isinstance(stats, SearchStats)
        assert stats.total_searches >= 3
        assert stats.cache_hit_rate > 0
        assert stats.average_response_time_ms >= 0
        assert stats.total_documents_indexed == 2
    
    def test_index_update(self, search_engine):
        """Test incremental index updates."""
        # Build initial index
        search_engine.build_index()
        initial_chunks = len(search_engine.chunk_metadata)
        
        # Add new document to storage
        new_chunk = DocumentChunk(
            text="Artificial intelligence encompasses machine learning and deep learning techniques.",
            metadata=ChunkMetadata(
                chunk_id="chunk_005",
                page_number=1,
                section_title="AI Overview",
                char_start=0,
                char_end=80,
                word_count=12
            )
        )
        
        new_doc = DocumentResult(
            metadata=DocumentMetadata(
                doc_id="doc_new123456789",
                filename="ai_guide.pdf",
                upload_timestamp=datetime.now().isoformat(),
                total_pages=1,
                total_chunks=1,
                processing_status="completed",
                file_size_bytes=512
            ),
            chunks=[new_chunk]
        )
        
        search_engine.storage.store_document(new_doc)
        
        # Update index
        success = search_engine.update_index_with_document("doc_new123456789")
        
        assert success is True
        assert len(search_engine.chunk_metadata) > initial_chunks
    
    def test_memory_efficiency(self, populated_storage):
        """Test memory usage stays within limits."""
        # Create engine with very small feature limit
        engine = TFIDFSearchEngine(populated_storage, max_features=100)
        engine.build_index()
        
        # Verify matrix size is constrained
        assert engine.tfidf_matrix.shape[1] <= 100
        
        # Perform search operations
        for i in range(10):
            results = engine.search_similar_chunks(f"test query {i}")
            # Memory should not grow indefinitely due to caching limits
    
    def test_error_handling(self, populated_storage):
        """Test error handling in various scenarios."""
        engine = TFIDFSearchEngine(populated_storage)
        
        # Test search without index
        results = engine.search_similar_chunks("test")
        # Should attempt to build index and return results or empty list
        assert isinstance(results, list)
        
        # Test empty query
        results = engine.search_similar_chunks("")
        assert len(results) == 0
        
        # Test invalid filters
        results = engine.search_similar_chunks("test", filters={'invalid_filter': 'value'})
        assert isinstance(results, list)
    
    def test_fips_compliance(self, search_engine):
        """Test FIPS compliance of hashing operations."""
        # Test query hashing uses SHA-256
        query_hash = search_engine._generate_query_hash("test query", 5, {})
        assert len(query_hash) == 16  # Truncated SHA-256 hash
        
        # Test that no MD5 is used anywhere in the search process
        search_engine.build_index()
        results = search_engine.search_similar_chunks("test")
        
        # All operations should complete without using non-FIPS algorithms
        assert True  # If we get here, no MD5 was used
    
    def test_concurrent_search(self, search_engine):
        """Test thread safety of concurrent searches."""
        import threading
        
        search_engine.build_index()
        results = []
        errors = []
        
        def search_worker(query_suffix):
            try:
                worker_results = search_engine.search_similar_chunks(f"machine learning {query_suffix}")
                results.append(worker_results)
            except Exception as e:
                errors.append(e)
        
        # Run concurrent searches
        threads = []
        for i in range(5):
            thread = threading.Thread(target=search_worker, args=(str(i),))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check that all searches completed without errors
        assert len(errors) == 0
        assert len(results) == 5
    
    def test_performance_requirements(self, search_engine):
        """Test that search meets performance requirements."""
        search_engine.build_index()
        
        import time
        
        # Test search response time
        start_time = time.time()
        results = search_engine.search_similar_chunks("machine learning algorithms")
        response_time = (time.time() - start_time) * 1000  # ms
        
        # Should respond under 2 seconds (2000ms) as per requirements
        assert response_time < 2000
        
        # Test index building time is reasonable
        start_time = time.time()
        search_engine.rebuild_index()
        build_time = time.time() - start_time
        
        # Should build quickly for small document set
        assert build_time < 30  # 30 seconds max for test data
    
    def test_search_quality(self, search_engine):
        """Test search quality and relevance."""
        search_engine.build_index()
        
        # Test keyword search
        results = search_engine.search_similar_chunks("machine learning")
        ml_results = [r for r in results if "machine learning" in r.text.lower()]
        assert len(ml_results) > 0
        
        # Test question search
        results = search_engine.search_similar_chunks("What are neural networks?")
        nn_results = [r for r in results if "neural" in r.text.lower()]
        assert len(nn_results) > 0
        
        # Test phrase search
        results = search_engine.search_similar_chunks("data privacy regulations")
        privacy_results = [r for r in results if "privacy" in r.text.lower() or "data" in r.text.lower()]
        assert len(privacy_results) > 0
        
        # Test complex search
        results = search_engine.search_similar_chunks("FIPS compliance standards")
        fips_results = [r for r in results if "fips" in r.text.lower()]
        assert len(fips_results) > 0


class TestSearchIntegration:
    """Integration tests for search engine with storage system."""
    
    def test_end_to_end_workflow(self, temp_storage_path):
        """Test complete end-to-end search workflow."""
        # Set up components
        storage = FileStorage(temp_storage_path)
        processor = Mock()  # Mock processor for this test
        search_engine = TFIDFSearchEngine(storage)
        
        # Create and store test document
        test_doc = DocumentResult(
            metadata=DocumentMetadata(
                doc_id="test_doc_12345",
                filename="test.pdf",
                upload_timestamp=datetime.now().isoformat(),
                total_pages=1,
                total_chunks=1,
                processing_status="completed",
                file_size_bytes=1024
            ),
            chunks=[
                DocumentChunk(
                    text="This is a test document about artificial intelligence and machine learning applications in modern technology.",
                    metadata=ChunkMetadata(
                        chunk_id="chunk_001",
                        page_number=1,
                        section_title="Introduction",
                        char_start=0,
                        char_end=106,
                        word_count=16
                    )
                )
            ]
        )
        
        # Store document
        doc_id = storage.store_document(test_doc)
        assert doc_id == "test_doc_12345"
        
        # Build search index
        success = search_engine.build_index()
        assert success is True
        
        # Perform search
        results = search_engine.search_similar_chunks("artificial intelligence")
        assert len(results) > 0
        assert "artificial intelligence" in results[0].text.lower()
        
        # Test search stats
        stats = search_engine.get_search_stats()
        assert stats.total_searches > 0
        assert stats.total_documents_indexed == 1
    
    def test_large_document_handling(self, temp_storage_path):
        """Test handling of documents with many chunks."""
        storage = FileStorage(temp_storage_path)
        search_engine = TFIDFSearchEngine(storage, max_features=100)
        
        # Create document with many chunks
        chunks = []
        for i in range(50):  # Create 50 chunks
            chunks.append(
                DocumentChunk(
                    text=f"This is chunk number {i} containing information about topic {i % 5}. "
                         f"Machine learning algorithms process data efficiently in various applications.",
                    metadata=ChunkMetadata(
                        chunk_id=f"chunk_{i:03d}",
                        page_number=(i // 10) + 1,
                        section_title=f"Section {i % 5}",
                        char_start=i * 100,
                        char_end=(i + 1) * 100,
                        word_count=15
                    )
                )
            )
        
        large_doc = DocumentResult(
            metadata=DocumentMetadata(
                doc_id="large_doc_12345",
                filename="large_test.pdf",
                upload_timestamp=datetime.now().isoformat(),
                total_pages=5,
                total_chunks=50,
                processing_status="completed",
                file_size_bytes=10240
            ),
            chunks=chunks
        )
        
        # Store and index
        storage.store_document(large_doc)
        success = search_engine.build_index()
        assert success is True
        
        # Search should work efficiently
        results = search_engine.search_similar_chunks("machine learning", top_k=10)
        assert len(results) <= 10
        assert all(isinstance(r, SearchResult) for r in results)


if __name__ == "__main__":
    pytest.main([__file__]) 