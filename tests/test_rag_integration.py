"""
Integration tests for complete RAG workflows.
Tests document-to-response pipeline, multiple document scenarios, and error recovery.
"""

import pytest
import tempfile
import shutil
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from rag.rag_pipeline import RAGPipeline
from rag.document_processor import DoclingProcessor, RAGConfig
from rag.file_storage import FileStorage
from rag.search_engine import TFIDFSearchEngine
from tests.fixtures.generate_test_data import TestDataGenerator, DOCUMENT_TEMPLATES


class TestRAGIntegration:
    """Test complete RAG integration workflows."""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def rag_config(self):
        """Create test RAG configuration."""
        return RAGConfig(
            max_memory_mb=50,
            chunk_size=800,
            chunk_overlap=100,
            max_search_results=5,
            relevance_threshold=0.1
        )
    
    @pytest.fixture
    def mock_ai_client(self):
        """Create mock AI client."""
        mock_client = Mock()
        mock_client.chat.return_value = iter([
            "Based on the provided context, machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed."
        ])
        return mock_client
    
    @pytest.fixture
    def rag_system(self, temp_storage_dir, rag_config, mock_ai_client):
        """Create complete RAG system for testing."""
        # Create storage
        storage = FileStorage(temp_storage_dir, max_memory_mb=rag_config.max_memory_mb)
        
        # Create search engine
        search_engine = TFIDFSearchEngine(storage, max_features=1000)
        
        # Create document processor
        with patch('rag.document_processor.DOCLING_AVAILABLE', True):
            with patch('rag.document_processor.DocumentConverter') as mock_converter:
                processor = DoclingProcessor(rag_config, storage=storage)
                
                # Mock the converter
                mock_doc = Mock()
                mock_doc.export_to_text.return_value = "Sample extracted text content"
                mock_result = Mock()
                mock_result.document = mock_doc
                mock_converter.return_value.convert.return_value = mock_result
        
        # Create RAG pipeline
        pipeline = RAGPipeline(
            storage=storage,
            search_engine=search_engine,
            ai_client=mock_ai_client,
            max_context_tokens=1000
        )
        
        return {
            'storage': storage,
            'search_engine': search_engine,
            'processor': processor,
            'pipeline': pipeline,
            'ai_client': mock_ai_client
        }
    
    def test_complete_rag_workflow(self, rag_system):
        """Test complete workflow from document to response."""
        storage = rag_system['storage']
        search_engine = rag_system['search_engine']
        processor = rag_system['processor']
        pipeline = rag_system['pipeline']
        
        # Step 1: Process and store a document
        test_content = """
        Machine Learning Fundamentals
        
        Machine learning is a subset of artificial intelligence (AI) that provides systems 
        the ability to automatically learn and improve from experience without being 
        explicitly programmed. Machine learning focuses on the development of computer 
        programs that can access data and use it to learn for themselves.
        
        Types of Machine Learning:
        1. Supervised Learning: Uses labeled training data
        2. Unsupervised Learning: Discovers patterns in unlabeled data
        3. Reinforcement Learning: Agent learns through interaction with environment
        """
        
        doc_result = processor.process_text(test_content, "ml_guide.txt")
        doc_id = storage.store_document(doc_result)
        
        # Verify document was stored
        assert doc_id in storage.document_index
        assert len(storage.document_index) == 1
        
        # Step 2: Build search index
        success = search_engine.build_index()
        assert success is True
        assert search_engine.tfidf_matrix is not None
        
        # Step 3: Process RAG query
        query = "What is machine learning?"
        response = pipeline.process_rag_query(query, model="test-model")
        
        # Verify response
        assert response is not None
        assert response.query == query
        assert response.rag_mode == "full"
        assert len(response.sources) > 0
        assert response.processing_time > 0
        assert "machine learning" in response.response.lower()
        
        # Verify pipeline stats
        assert pipeline.stats['total_queries'] == 1
        assert pipeline.stats['full_rag_queries'] == 1
    
    def test_multiple_documents_scenario(self, rag_system):
        """Test RAG with multiple documents from different domains."""
        storage = rag_system['storage']
        search_engine = rag_system['search_engine']
        processor = rag_system['processor']
        pipeline = rag_system['pipeline']
        
        # Add multiple documents with different content
        documents = [
            {
                "content": """
                Machine Learning Guide
                
                Machine learning algorithms can be categorized into supervised, unsupervised, 
                and reinforcement learning. Supervised learning uses labeled data to train 
                models for classification and regression tasks.
                """,
                "filename": "ml_guide.txt"
            },
            {
                "content": """
                Data Science Handbook
                
                Data science involves collecting, cleaning, and analyzing data to extract 
                insights. The data science pipeline includes data collection, preprocessing, 
                analysis, modeling, and visualization.
                """,
                "filename": "ds_handbook.txt"
            },
            {
                "content": """
                Security Standards Documentation
                
                FIPS 140-2 is a U.S. government standard for cryptographic modules. 
                It specifies security requirements for cryptographic modules used in 
                government systems. SHA-256 is a FIPS-approved hash function.
                """,
                "filename": "security_doc.txt"
            }
        ]
        
        # Process and store all documents
        doc_ids = []
        for doc in documents:
            doc_result = processor.process_text(doc["content"], doc["filename"])
            doc_id = storage.store_document(doc_result)
            doc_ids.append(doc_id)
        
        # Build search index
        search_engine.build_index()
        
        # Test queries across different domains
        test_cases = [
            {
                "query": "What is supervised learning?",
                "expected_domains": ["machine learning", "supervised"],
                "expected_sources": 1
            },
            {
                "query": "Describe the data science pipeline",
                "expected_domains": ["data science", "pipeline"],
                "expected_sources": 1
            },
            {
                "query": "What are FIPS standards?",
                "expected_domains": ["FIPS", "cryptographic", "security"],
                "expected_sources": 1
            },
            {
                "query": "How does machine learning relate to data science?",
                "expected_domains": ["machine learning", "data science"],
                "expected_sources": 2  # Should match both domains
            }
        ]
        
        for test_case in test_cases:
            response = pipeline.process_rag_query(test_case["query"], model="test-model")
            
            # Verify response quality
            assert response.rag_mode == "full"
            assert len(response.sources) >= test_case["expected_sources"]
            
            # Check that response contains expected domain terms
            response_lower = response.response.lower()
            domain_matches = sum(1 for domain in test_case["expected_domains"] 
                               if domain.lower() in response_lower)
            assert domain_matches > 0, f"Response should contain terms from {test_case['expected_domains']}"
    
    def test_memory_constraints_integration(self, temp_storage_dir, mock_ai_client):
        """Test system behavior under memory constraints."""
        # Create system with very low memory limit
        low_memory_config = RAGConfig(max_memory_mb=1)  # Very low limit
        
        storage = FileStorage(temp_storage_dir, max_memory_mb=1)
        search_engine = TFIDFSearchEngine(storage, max_features=100)
        
        with patch('rag.document_processor.DOCLING_AVAILABLE', True):
            with patch('rag.document_processor.DocumentConverter'):
                processor = DoclingProcessor(low_memory_config, storage=storage)
        
        pipeline = RAGPipeline(storage, search_engine, mock_ai_client)
        
        # Add multiple documents to trigger memory management
        for i in range(5):
            large_content = f"Document {i} content. " * 100  # Large content
            doc_result = processor.process_text(large_content, f"doc_{i}.txt")
            storage.store_document(doc_result)
        
        # Build index - should handle memory constraints
        success = search_engine.build_index()
        assert success is True
        
        # Verify memory usage is managed
        memory_stats = storage.memory_manager.get_cache_stats()
        assert memory_stats['current_memory_mb'] <= 1.5  # Some tolerance
        
        # System should still be able to process queries
        response = pipeline.process_rag_query("What is this about?", model="test-model")
        assert response is not None
    
    def test_concurrent_queries(self, rag_system):
        """Test handling of concurrent queries."""
        import threading
        
        storage = rag_system['storage']
        search_engine = rag_system['search_engine']
        processor = rag_system['processor']
        pipeline = rag_system['pipeline']
        
        # Set up test document
        test_content = "This is test content for concurrent access testing."
        doc_result = processor.process_text(test_content, "concurrent_test.txt")
        storage.store_document(doc_result)
        search_engine.build_index()
        
        # Results storage
        results = {}
        errors = {}
        
        def query_worker(worker_id: int):
            """Worker function for concurrent queries."""
            try:
                response = pipeline.process_rag_query(
                    f"Query from worker {worker_id}", 
                    model="test-model"
                )
                results[worker_id] = response
            except Exception as e:
                errors[worker_id] = str(e)
        
        # Launch concurrent workers
        threads = []
        num_workers = 5
        
        for i in range(num_workers):
            thread = threading.Thread(target=query_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all workers to complete
        for thread in threads:
            thread.join()
        
        # Verify all queries completed successfully
        assert len(errors) == 0, f"Concurrent query errors: {errors}"
        assert len(results) == num_workers
        
        # Verify all responses are valid
        for worker_id, response in results.items():
            assert response is not None
            assert response.query == f"Query from worker {worker_id}"
            assert response.processing_time > 0
    
    def test_error_recovery_and_graceful_degradation(self, rag_system):
        """Test system behavior with various error conditions."""
        storage = rag_system['storage']
        search_engine = rag_system['search_engine']
        processor = rag_system['processor']
        pipeline = rag_system['pipeline']
        
        # Test 1: Query with no documents
        response = pipeline.process_rag_query("What is this about?", model="test-model")
        assert response.rag_mode == "fallback"
        assert len(response.sources) == 0
        
        # Test 2: Add document and test normal operation
        doc_result = processor.process_text("Test content", "test.txt")
        storage.store_document(doc_result)
        search_engine.build_index()
        
        response = pipeline.process_rag_query("What is this about?", model="test-model")
        assert response.rag_mode == "full"
        
        # Test 3: Simulate search engine failure
        with patch.object(search_engine, 'search_similar_chunks', side_effect=Exception("Search failed")):
            response = pipeline.process_rag_query("What is this about?", model="test-model")
            assert response.rag_mode == "fallback"
        
        # Test 4: Simulate AI client failure
        with patch.object(pipeline.ai_client, 'chat', side_effect=Exception("AI client failed")):
            with pytest.raises(Exception):
                pipeline.process_rag_query("What is this about?", model="test-model")
    
    def test_document_update_and_index_refresh(self, rag_system):
        """Test document updates and index refreshing."""
        storage = rag_system['storage']
        search_engine = rag_system['search_engine']
        processor = rag_system['processor']
        pipeline = rag_system['pipeline']
        
        # Initial document
        initial_content = "Initial document content about machine learning."
        doc_result = processor.process_text(initial_content, "test_doc.txt")
        doc_id = storage.store_document(doc_result)
        search_engine.build_index()
        
        # Query should find the document
        response = pipeline.process_rag_query("machine learning", model="test-model")
        assert response.rag_mode == "full"
        initial_source_count = len(response.sources)
        
        # Add more documents
        for i in range(3):
            new_content = f"Additional document {i} about data science and analytics."
            new_doc = processor.process_text(new_content, f"doc_{i}.txt")
            storage.store_document(new_doc)
        
        # Index should be updated automatically or manually
        search_engine.build_index()
        
        # Query should now find more relevant documents
        response = pipeline.process_rag_query("data science", model="test-model")
        assert response.rag_mode == "full"
        assert len(response.sources) >= 1
        
        # Verify search statistics are updated
        stats = search_engine.get_search_stats()
        assert stats.total_documents_indexed == 4
    
    def test_large_query_context_handling(self, rag_system):
        """Test handling of queries that generate large context."""
        storage = rag_system['storage']
        search_engine = rag_system['search_engine']
        processor = rag_system['processor']
        pipeline = rag_system['pipeline']
        
        # Create documents with long content
        long_content = """
        Machine Learning Comprehensive Guide
        
        """ + "This is detailed content about machine learning. " * 200
        
        doc_result = processor.process_text(long_content, "long_doc.txt")
        storage.store_document(doc_result)
        search_engine.build_index()
        
        # Query that might generate large context
        response = pipeline.process_rag_query(
            "Tell me everything about machine learning",
            model="test-model"
        )
        
        # Should handle large context gracefully
        assert response.rag_mode == "full"
        assert len(response.context_used) > 0
        assert response.processing_time > 0
        
        # Context should be within reasonable limits
        context_length = len(response.context_used)
        assert context_length < 10000  # Reasonable upper limit
    
    def test_edge_case_queries(self, rag_system):
        """Test edge case queries and inputs."""
        storage = rag_system['storage']
        search_engine = rag_system['search_engine']
        processor = rag_system['processor']
        pipeline = rag_system['pipeline']
        
        # Set up test document
        doc_result = processor.process_text("Test document content", "test.txt")
        storage.store_document(doc_result)
        search_engine.build_index()
        
        edge_cases = [
            "",  # Empty query
            "   ",  # Whitespace only
            "a" * 1000,  # Very long query
            "!@#$%^&*()",  # Special characters only
            "What is the meaning of life, the universe, and everything?",  # Philosophical query
            "машинное обучение",  # Non-English query
        ]
        
        for query in edge_cases:
            response = pipeline.process_rag_query(query, model="test-model")
            
            # Should not crash and should return valid response
            assert response is not None
            assert hasattr(response, 'rag_mode')
            assert hasattr(response, 'processing_time')
            assert response.processing_time >= 0
    
    def test_performance_benchmarks(self, rag_system):
        """Test performance benchmarks for the integration."""
        storage = rag_system['storage']
        search_engine = rag_system['search_engine']
        processor = rag_system['processor']
        pipeline = rag_system['pipeline']
        
        # Set up multiple test documents
        for i in range(10):
            content = f"Document {i} content with various topics like machine learning, data science, and AI."
            doc_result = processor.process_text(content, f"doc_{i}.txt")
            storage.store_document(doc_result)
        
        # Build index and measure time
        start_time = time.time()
        search_engine.build_index()
        index_time = time.time() - start_time
        
        # Index building should be reasonably fast
        assert index_time < 10.0, f"Index building took too long: {index_time}s"
        
        # Test query performance
        queries = [
            "What is machine learning?",
            "Explain data science",
            "Tell me about AI"
        ]
        
        query_times = []
        for query in queries:
            start_time = time.time()
            response = pipeline.process_rag_query(query, model="test-model")
            query_time = time.time() - start_time
            query_times.append(query_time)
            
            # Each query should complete within reasonable time
            assert query_time < 5.0, f"Query took too long: {query_time}s"
            assert response.rag_mode == "full"
        
        # Average query time should be reasonable
        avg_query_time = sum(query_times) / len(query_times)
        assert avg_query_time < 3.0, f"Average query time too high: {avg_query_time}s"


class TestRAGIntegrationWithRealData:
    """Test RAG integration with realistic test data."""
    
    @pytest.fixture
    def test_data_dir(self):
        """Create test data directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_ai_client(self):
        """Create mock AI client with realistic responses."""
        mock_client = Mock()
        
        def mock_chat_response(messages, **kwargs):
            # Generate response based on the context in messages
            context = ""
            for message in messages:
                if isinstance(message, dict) and message.get('role') == 'user':
                    context += message.get('content', '')
            
            if "machine learning" in context.lower():
                return iter(["Machine learning is a subset of AI that enables computers to learn from data."])
            elif "data science" in context.lower():
                return iter(["Data science involves extracting insights from data using statistical methods."])
            elif "fips" in context.lower():
                return iter(["FIPS standards ensure cryptographic security in government systems."])
            else:
                return iter(["I can provide information based on the available documents."])
        
        mock_client.chat = mock_chat_response
        return mock_client
    
    def test_integration_with_generated_test_data(self, test_data_dir, mock_ai_client):
        """Test RAG integration with generated test documents."""
        # Generate test data
        generator = TestDataGenerator(test_data_dir)
        
        # Create a smaller test set for integration testing
        test_docs = []
        for template_key in DOCUMENT_TEMPLATES.keys():
            pdf_path = generator.generate_sample_pdf(
                template_key, 
                f"test_{template_key}.pdf",
                pages_target=2
            )
            test_docs.append(pdf_path)
        
        # Set up RAG system
        config = RAGConfig(max_memory_mb=50)
        storage = FileStorage(test_data_dir, max_memory_mb=50)
        search_engine = TFIDFSearchEngine(storage)
        
        with patch('rag.document_processor.DOCLING_AVAILABLE', True):
            with patch('rag.document_processor.DocumentConverter') as mock_converter:
                # Mock PDF processing
                mock_doc = Mock()
                mock_doc.export_to_text.return_value = "Extracted PDF content about the topic"
                mock_result = Mock()
                mock_result.document = mock_doc
                mock_converter.return_value.convert.return_value = mock_result
                
                processor = DoclingProcessor(config, storage=storage)
        
        pipeline = RAGPipeline(storage, search_engine, mock_ai_client)
        
        # Process all test documents
        for pdf_path in test_docs:
            doc_result = processor.process_pdf(pdf_path)
            storage.store_document(doc_result)
        
        # Build search index
        search_engine.build_index()
        
        # Test queries from generated test data
        test_queries = generator.generate_test_queries()
        
        for query_data in test_queries[:3]:  # Test first 3 queries
            query = query_data["query"]
            expected_topics = query_data["expected_topics"]
            
            response = pipeline.process_rag_query(query, model="test-model")
            
            # Verify response quality
            assert response.rag_mode == "full"
            assert len(response.sources) > 0
            assert response.processing_time > 0
            
            # Check if response contains expected topics
            response_lower = response.response.lower()
            topic_matches = sum(1 for topic in expected_topics 
                              if topic.lower() in response_lower)
            
            # At least one expected topic should be mentioned
            assert topic_matches > 0, f"Response should mention topics: {expected_topics}" 