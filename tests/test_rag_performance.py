"""
Performance tests for RAG system.
Tests memory usage, response times, concurrent access, and large document processing.
"""

import pytest
import psutil
import time
import threading
import tempfile
import shutil
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import Mock, patch
from typing import List, Dict, Any, Tuple

from rag.rag_pipeline import RAGPipeline
from rag.document_processor import DoclingProcessor, RAGConfig
from rag.file_storage import FileStorage
from rag.search_engine import TFIDFSearchEngine
from tests.fixtures.generate_test_data import TestDataGenerator


class PerformanceMonitor:
    """Utility class for monitoring system performance during tests."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_mb()
        self.memory_samples = []
        self.monitoring = False
        self.monitor_thread = None
    
    def get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_cpu_percent(self) -> float:
        """Get current CPU usage percentage."""
        return self.process.cpu_percent()
    
    def start_monitoring(self, interval: float = 0.1):
        """Start continuous memory monitoring."""
        self.monitoring = True
        self.memory_samples = []
        
        def monitor():
            while self.monitoring:
                self.memory_samples.append(self.get_memory_mb())
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return statistics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        if not self.memory_samples:
            return {}
        
        return {
            'initial_memory_mb': self.initial_memory,
            'final_memory_mb': self.memory_samples[-1],
            'peak_memory_mb': max(self.memory_samples),
            'min_memory_mb': min(self.memory_samples),
            'avg_memory_mb': statistics.mean(self.memory_samples),
            'memory_increase_mb': self.memory_samples[-1] - self.initial_memory,
            'peak_increase_mb': max(self.memory_samples) - self.initial_memory
        }


class TestRAGPerformance:
    """Test RAG system performance characteristics."""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def performance_config(self):
        """Create configuration optimized for performance testing."""
        return RAGConfig(
            max_memory_mb=100,
            chunk_size=1000,
            chunk_overlap=200,
            max_search_results=10,
            relevance_threshold=0.1,
            cache_size=50
        )
    
    @pytest.fixture
    def mock_ai_client(self):
        """Create mock AI client with controlled response times."""
        mock_client = Mock()
        
        def mock_chat_with_delay(messages, **kwargs):
            # Simulate AI response time
            time.sleep(0.1)  # 100ms simulated response time
            return iter(["This is a test response based on the provided context."])
        
        mock_client.chat = mock_chat_with_delay
        return mock_client
    
    @pytest.fixture
    def rag_system(self, temp_storage_dir, performance_config, mock_ai_client):
        """Create RAG system for performance testing."""
        storage = FileStorage(temp_storage_dir, max_memory_mb=performance_config.max_memory_mb)
        search_engine = TFIDFSearchEngine(storage, max_features=2000)
        
        with patch('rag.document_processor.DOCLING_AVAILABLE', True):
            with patch('rag.document_processor.DocumentConverter') as mock_converter:
                processor = DoclingProcessor(performance_config, storage=storage)
                
                # Mock converter with realistic processing time
                def mock_convert_with_delay(pdf_path):
                    time.sleep(0.05)  # 50ms simulated processing time
                    mock_doc = Mock()
                    mock_doc.export_to_text.return_value = f"Extracted content from {Path(pdf_path).name}"
                    mock_result = Mock()
                    mock_result.document = mock_doc
                    return mock_result
                
                mock_converter.return_value.convert.side_effect = mock_convert_with_delay
        
        pipeline = RAGPipeline(storage, search_engine, mock_ai_client)
        
        return {
            'storage': storage,
            'search_engine': search_engine,
            'processor': processor,
            'pipeline': pipeline,
            'config': performance_config
        }
    
    def test_memory_usage_within_limits(self, rag_system):
        """Test that RAG system stays within specified memory limits."""
        storage = rag_system['storage']
        processor = rag_system['processor']
        search_engine = rag_system['search_engine']
        config = rag_system['config']
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        try:
            # Process target number of documents for demo scenario
            for i in range(20):  # Subset of demo target (75 docs)
                # Create document with realistic size
                content = f"Document {i} content. " * 200  # ~2KB per document
                doc_result = processor.process_text(content, f"doc_{i}.txt")
                storage.store_document(doc_result)
            
            # Build search index
            search_engine.build_index()
            
            # Perform multiple queries to test sustained memory usage
            for j in range(10):
                response = rag_system['pipeline'].process_rag_query(
                    f"Query {j} about the documents", 
                    model="test-model"
                )
                assert response is not None
            
        finally:
            stats = monitor.stop_monitoring()
        
        # Verify memory constraints
        assert stats['peak_increase_mb'] < config.max_memory_mb, \
            f"Peak memory increase ({stats['peak_increase_mb']:.1f}MB) exceeded limit ({config.max_memory_mb}MB)"
        
        # Memory should not continuously grow (no significant leaks)
        if len(monitor.memory_samples) > 10:
            early_avg = statistics.mean(monitor.memory_samples[:5])
            late_avg = statistics.mean(monitor.memory_samples[-5:])
            growth_rate = (late_avg - early_avg) / early_avg
            assert growth_rate < 0.2, f"Memory growth rate too high: {growth_rate:.2%}"
    
    def test_query_response_time_requirements(self, rag_system):
        """Test that RAG queries respond within required time limits."""
        storage = rag_system['storage']
        processor = rag_system['processor']
        search_engine = rag_system['search_engine']
        pipeline = rag_system['pipeline']
        
        # Set up test documents
        for i in range(10):
            content = f"""
            Document {i} about machine learning and artificial intelligence.
            This document contains information about various AI topics including
            neural networks, deep learning, natural language processing, and computer vision.
            The content is designed to provide relevant matches for test queries.
            """
            doc_result = processor.process_text(content, f"ai_doc_{i}.txt")
            storage.store_document(doc_result)
        
        search_engine.build_index()
        
        # Test queries with different complexity levels
        test_queries = [
            "What is machine learning?",  # Simple query
            "Explain the relationship between AI and neural networks",  # Complex query
            "How does deep learning differ from traditional machine learning approaches?",  # Very complex
            "AI",  # Very short query
            "artificial intelligence machine learning deep learning neural networks"  # Long query
        ]
        
        response_times = []
        
        for query in test_queries:
            start_time = time.time()
            response = pipeline.process_rag_query(query, model="test-model")
            response_time = time.time() - start_time
            response_times.append(response_time)
            
            # Each query should complete within 5 seconds (requirement from guide)
            assert response_time < 5.0, f"Query '{query}' took too long: {response_time:.2f}s"
            assert response.rag_mode == "full"
        
        # Average response time should be reasonable
        avg_response_time = statistics.mean(response_times)
        assert avg_response_time < 2.0, f"Average response time too high: {avg_response_time:.2f}s"
        
        # 95th percentile should be under threshold
        p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
        assert p95_response_time < 3.0, f"95th percentile response time too high: {p95_response_time:.2f}s"
    
    def test_concurrent_user_performance(self, rag_system):
        """Test system performance with multiple concurrent users."""
        storage = rag_system['storage']
        processor = rag_system['processor']
        search_engine = rag_system['search_engine']
        pipeline = rag_system['pipeline']
        
        # Set up test documents
        for i in range(15):
            content = f"Shared document {i} with content about technology and innovation."
            doc_result = processor.process_text(content, f"shared_doc_{i}.txt")
            storage.store_document(doc_result)
        
        search_engine.build_index()
        
        # Define concurrent user simulation
        def user_session(user_id: int, num_queries: int = 3) -> List[float]:
            """Simulate a user session with multiple queries."""
            response_times = []
            queries = [
                f"User {user_id} query about technology",
                f"What does user {user_id} want to know about innovation?",
                f"User {user_id} asks about the documents"
            ]
            
            for i in range(num_queries):
                query = queries[i % len(queries)]
                start_time = time.time()
                
                try:
                    response = pipeline.process_rag_query(query, model="test-model")
                    response_time = time.time() - start_time
                    response_times.append(response_time)
                    
                    # Verify response quality under load
                    assert response is not None
                    assert response.rag_mode in ["full", "partial", "fallback"]
                    
                except Exception as e:
                    response_times.append(float('inf'))  # Mark as failed
                    print(f"User {user_id} query failed: {e}")
            
            return response_times
        
        # Test with increasing numbers of concurrent users
        user_counts = [1, 3, 5, 8]
        results = {}
        
        monitor = PerformanceMonitor()
        
        for num_users in user_counts:
            monitor.start_monitoring()
            start_time = time.time()
            
            # Run concurrent user sessions
            with ThreadPoolExecutor(max_workers=num_users) as executor:
                futures = [
                    executor.submit(user_session, user_id, 2)
                    for user_id in range(num_users)
                ]
                
                user_results = []
                for future in as_completed(futures):
                    user_results.extend(future.result())
            
            total_time = time.time() - start_time
            memory_stats = monitor.stop_monitoring()
            
            # Filter out failed requests
            successful_times = [t for t in user_results if t != float('inf')]
            
            results[num_users] = {
                'total_time': total_time,
                'successful_requests': len(successful_times),
                'failed_requests': len(user_results) - len(successful_times),
                'avg_response_time': statistics.mean(successful_times) if successful_times else 0,
                'max_response_time': max(successful_times) if successful_times else 0,
                'memory_peak_mb': memory_stats.get('peak_memory_mb', 0),
                'memory_increase_mb': memory_stats.get('memory_increase_mb', 0)
            }
        
        # Verify performance requirements
        for num_users, metrics in results.items():
            # All requests should succeed
            assert metrics['failed_requests'] == 0, \
                f"Failed requests with {num_users} users: {metrics['failed_requests']}"
            
            # Response times should remain reasonable under load
            assert metrics['avg_response_time'] < 10.0, \
                f"Average response time too high with {num_users} users: {metrics['avg_response_time']:.2f}s"
            
            # Memory usage should not explode under load
            assert metrics['memory_increase_mb'] < 150, \
                f"Memory usage too high with {num_users} users: {metrics['memory_increase_mb']:.1f}MB"
        
        # Performance should degrade gracefully with load
        single_user_avg = results[1]['avg_response_time']
        multi_user_avg = results[max(user_counts)]['avg_response_time']
        degradation_ratio = multi_user_avg / single_user_avg if single_user_avg > 0 else 1
        
        assert degradation_ratio < 3.0, \
            f"Performance degradation too severe: {degradation_ratio:.1f}x slower"
    
    def test_large_document_set_performance(self, temp_storage_dir, performance_config, mock_ai_client):
        """Test performance with large document sets (demo target: 75 PDFs, 300 pages)."""
        # Create system optimized for large document sets
        storage = FileStorage(temp_storage_dir, max_memory_mb=performance_config.max_memory_mb)
        search_engine = TFIDFSearchEngine(storage, max_features=5000)  # Higher feature count
        
        with patch('rag.document_processor.DOCLING_AVAILABLE', True):
            with patch('rag.document_processor.DocumentConverter') as mock_converter:
                processor = DoclingProcessor(performance_config, storage=storage)
                
                mock_doc = Mock()
                mock_doc.export_to_text.return_value = "Large document content with comprehensive information."
                mock_result = Mock()
                mock_result.document = mock_doc
                mock_converter.return_value.convert.return_value = mock_result
        
        pipeline = RAGPipeline(storage, search_engine, mock_ai_client)
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        try:
            # Simulate demo document set (scaled down for test performance)
            num_docs = 25  # Scaled down from 75 for faster testing
            target_pages = 100  # Scaled down from 300
            pages_per_doc = target_pages // num_docs
            
            # Process documents with realistic content size
            processing_times = []
            
            for i in range(num_docs):
                # Create content equivalent to multiple pages
                page_content = "This is page content with detailed information. " * 100
                full_content = page_content * pages_per_doc
                
                start_time = time.time()
                doc_result = processor.process_text(full_content, f"large_doc_{i:02d}.pdf")
                storage.store_document(doc_result)
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                # Progress check for large sets
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{num_docs} documents...")
            
            # Build search index for large document set
            print("Building search index...")
            index_start_time = time.time()
            success = search_engine.build_index()
            index_build_time = time.time() - index_start_time
            
            assert success is True, "Failed to build index for large document set"
            
            # Test search performance on large index
            search_queries = [
                "comprehensive information",
                "detailed content",
                "page content information",
                "document with content"
            ]
            
            search_times = []
            for query in search_queries:
                start_time = time.time()
                response = pipeline.process_rag_query(query, model="test-model")
                search_time = time.time() - start_time
                search_times.append(search_time)
                
                assert response.rag_mode == "full"
                assert len(response.sources) > 0
        
        finally:
            memory_stats = monitor.stop_monitoring()
        
        # Performance assertions for large document sets
        avg_processing_time = statistics.mean(processing_times)
        assert avg_processing_time < 1.0, \
            f"Average document processing time too high: {avg_processing_time:.2f}s"
        
        assert index_build_time < 30.0, \
            f"Index building took too long: {index_build_time:.1f}s"
        
        avg_search_time = statistics.mean(search_times)
        assert avg_search_time < 5.0, \
            f"Average search time too high: {avg_search_time:.2f}s"
        
        # Memory usage should stay within bounds
        assert memory_stats['peak_increase_mb'] < performance_config.max_memory_mb, \
            f"Memory usage exceeded limit: {memory_stats['peak_increase_mb']:.1f}MB"
        
        # Verify system can handle the scale
        total_chunks = sum(len(storage.load_document(doc_id).chunks) 
                          for doc_id in storage.document_index.keys())
        assert total_chunks > 100, "Should have generated substantial number of chunks"
        
        print(f"Large document set test completed:")
        print(f"  Documents processed: {num_docs}")
        print(f"  Total chunks: {total_chunks}")
        print(f"  Index build time: {index_build_time:.1f}s")
        print(f"  Average search time: {avg_search_time:.2f}s")
        print(f"  Peak memory increase: {memory_stats['peak_increase_mb']:.1f}MB")
    
    def test_memory_leak_detection(self, rag_system):
        """Test for memory leaks during sustained operation."""
        storage = rag_system['storage']
        processor = rag_system['processor']
        search_engine = rag_system['search_engine']
        pipeline = rag_system['pipeline']
        
        # Set up initial documents
        for i in range(5):
            content = f"Memory test document {i} with standard content."
            doc_result = processor.process_text(content, f"memory_test_{i}.txt")
            storage.store_document(doc_result)
        
        search_engine.build_index()
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring(interval=0.2)  # More frequent sampling
        
        try:
            # Perform sustained operations
            for cycle in range(50):  # Many operations to detect leaks
                # Mix of operations
                if cycle % 10 == 0:
                    # Add new document occasionally
                    content = f"Cycle {cycle} document with new content."
                    doc_result = processor.process_text(content, f"cycle_{cycle}.txt")
                    storage.store_document(doc_result)
                    
                    if cycle % 20 == 0:
                        # Rebuild index occasionally
                        search_engine.build_index()
                
                # Perform query
                response = pipeline.process_rag_query(
                    f"Query {cycle} about the test documents",
                    model="test-model"
                )
                assert response is not None
                
                # Force garbage collection occasionally
                if cycle % 15 == 0:
                    import gc
                    gc.collect()
        
        finally:
            memory_stats = monitor.stop_monitoring()
        
        # Analyze memory usage pattern for leaks
        if len(monitor.memory_samples) > 20:
            # Check if memory usage shows concerning upward trend
            early_samples = monitor.memory_samples[:10]
            late_samples = monitor.memory_samples[-10:]
            
            early_avg = statistics.mean(early_samples)
            late_avg = statistics.mean(late_samples)
            
            memory_growth = late_avg - early_avg
            growth_percentage = (memory_growth / early_avg) * 100
            
            # Allow some growth but flag significant increases
            assert growth_percentage < 50, \
                f"Potential memory leak detected: {growth_percentage:.1f}% growth"
            
            # Check for continuous growth (all late samples > all early samples)
            max_early = max(early_samples)
            min_late = min(late_samples)
            
            if min_late > max_early:
                # This could indicate a leak, but allow for some variance
                assert memory_growth < 20, \
                    f"Continuous memory growth detected: {memory_growth:.1f}MB"
    
    def test_search_performance_benchmarks(self, rag_system):
        """Test search engine performance with various query patterns."""
        storage = rag_system['storage']
        processor = rag_system['processor']
        search_engine = rag_system['search_engine']
        
        # Create diverse document set for search testing
        document_types = [
            ("Technical documentation about machine learning algorithms and neural networks.", "technical"),
            ("Business report on quarterly performance and financial metrics.", "business"),
            ("Scientific paper about quantum computing and physics principles.", "scientific"),
            ("User manual for software installation and configuration procedures.", "manual"),
            ("News article about technology trends and industry developments.", "news")
        ]
        
        # Create multiple documents of each type
        for doc_type, category in document_types:
            for i in range(4):  # 4 docs per type
                expanded_content = f"{doc_type} " * 50  # Expand content
                doc_result = processor.process_text(
                    expanded_content, 
                    f"{category}_doc_{i}.txt"
                )
                storage.store_document(doc_result)
        
        # Build index and measure time
        index_start = time.time()
        search_engine.build_index()
        index_time = time.time() - index_start
        
        # Test different search patterns
        search_patterns = [
            ("machine learning", "exact_phrase"),
            ("neural networks algorithms", "multi_term"),
            ("quantum", "single_term"),
            ("performance metrics financial", "cross_category"),
            ("installation configuration manual", "domain_specific"),
            ("technology trends industry developments", "long_query"),
            ("xyz123nonexistent", "no_matches")
        ]
        
        search_results = {}
        
        for query, pattern_type in search_patterns:
            # Measure search time
            start_time = time.time()
            results = search_engine.search_similar_chunks(query, top_k=5)
            search_time = time.time() - start_time
            
            search_results[pattern_type] = {
                'query': query,
                'search_time': search_time,
                'result_count': len(results),
                'top_score': results[0].similarity_score if results else 0
            }
        
        # Performance assertions
        assert index_time < 5.0, f"Index build time too high: {index_time:.2f}s"
        
        for pattern_type, metrics in search_results.items():
            assert metrics['search_time'] < 1.0, \
                f"Search time too high for {pattern_type}: {metrics['search_time']:.3f}s"
        
        # Verify search quality
        assert search_results['exact_phrase']['result_count'] > 0, "Should find exact phrase matches"
        assert search_results['exact_phrase']['top_score'] > 0.3, "Exact phrase should have high relevance"
        
        # No matches case should be fast
        assert search_results['no_matches']['search_time'] < 0.1, "No-match queries should be very fast"
        assert search_results['no_matches']['result_count'] == 0, "Should return no results for non-existent terms"
        
        print("Search performance benchmark results:")
        for pattern_type, metrics in search_results.items():
            print(f"  {pattern_type}: {metrics['search_time']:.3f}s, "
                  f"{metrics['result_count']} results, "
                  f"top score: {metrics['top_score']:.3f}")
    
    def test_system_resource_efficiency(self, rag_system):
        """Test overall system resource efficiency."""
        storage = rag_system['storage']
        processor = rag_system['processor']
        search_engine = rag_system['search_engine']
        pipeline = rag_system['pipeline']
        
        # Monitor CPU and memory during typical workload
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        cpu_samples = []
        
        try:
            # Simulate typical usage pattern
            workload_phases = [
                ("document_processing", 5),
                ("index_building", 1),
                ("query_processing", 10)
            ]
            
            for phase, iterations in workload_phases:
                phase_start = time.time()
                
                if phase == "document_processing":
                    for i in range(iterations):
                        content = f"Resource efficiency test document {i}."
                        doc_result = processor.process_text(content, f"efficiency_{i}.txt")
                        storage.store_document(doc_result)
                        cpu_samples.append(monitor.get_cpu_percent())
                
                elif phase == "index_building":
                    search_engine.build_index()
                    cpu_samples.append(monitor.get_cpu_percent())
                
                elif phase == "query_processing":
                    for i in range(iterations):
                        response = pipeline.process_rag_query(
                            f"Efficiency query {i}",
                            model="test-model"
                        )
                        cpu_samples.append(monitor.get_cpu_percent())
                
                phase_time = time.time() - phase_start
                print(f"{phase} phase completed in {phase_time:.2f}s")
        
        finally:
            memory_stats = monitor.stop_monitoring()
        
        # Resource efficiency assertions
        if cpu_samples:
            avg_cpu = statistics.mean(cpu_samples)
            max_cpu = max(cpu_samples)
            
            # CPU usage should be reasonable (not constantly at 100%)
            assert avg_cpu < 80, f"Average CPU usage too high: {avg_cpu:.1f}%"
            assert max_cpu < 95, f"Peak CPU usage too high: {max_cpu:.1f}%"
        
        # Memory efficiency
        assert memory_stats['peak_increase_mb'] < 100, \
            f"Memory usage too high: {memory_stats['peak_increase_mb']:.1f}MB"
        
        # Memory should be released after operations
        final_increase = memory_stats['memory_increase_mb']
        peak_increase = memory_stats['peak_increase_mb']
        memory_released = peak_increase - final_increase
        
        if peak_increase > 10:  # Only check if we used significant memory
            release_percentage = (memory_released / peak_increase) * 100
            assert release_percentage > 20, \
                f"Insufficient memory release: only {release_percentage:.1f}% released"
        
        print(f"Resource efficiency results:")
        print(f"  Average CPU: {statistics.mean(cpu_samples):.1f}%" if cpu_samples else "  CPU: Not measured")
        print(f"  Peak memory increase: {memory_stats['peak_increase_mb']:.1f}MB")
        print(f"  Memory released: {memory_released:.1f}MB ({release_percentage if 'release_percentage' in locals() else 0:.1f}%)")


if __name__ == "__main__":
    # Allow running individual performance tests
    pytest.main([__file__, "-v", "-s"]) 