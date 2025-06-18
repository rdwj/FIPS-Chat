# Chat Guide 07: Testing & Validation

## Objective
Create comprehensive testing suite for the RAG system to ensure reliability, performance, and FIPS compliance. This includes unit tests, integration tests, performance tests, and validation of the complete workflow.

## Prerequisites
- Chat Guide 01-06 completed (full RAG system with configuration)
- All RAG components implemented and functional
- Configuration management working
- Feature branch: `feature/rag-implementation`

## Goals for This Chat Session
1. Create comprehensive unit tests for all RAG components
2. Build integration tests for end-to-end RAG workflow
3. Implement performance and memory usage tests
4. Add FIPS compliance validation tests
5. Create test data and fixtures for consistent testing
6. Build demo validation tests for the target use case

## Tasks to Complete

### 1. Unit Test Suite
**What to Ask For:**
```
Create comprehensive unit tests:
- tests/test_rag_document_processor.py (Docling integration, chunking)
- tests/test_rag_file_storage.py (storage operations, memory management)
- tests/test_rag_search_engine.py (TF-IDF search, relevance scoring)
- tests/test_rag_pipeline.py (end-to-end pipeline, context management)
- tests/test_rag_config.py (configuration loading, validation)
Mock external dependencies and test error conditions
```

### 2. Integration Tests
**What to Ask For:**
```
Create integration tests for complete workflows:
- tests/test_rag_integration.py with full document-to-response testing
- Test document upload through RAG response generation
- Test multiple document scenarios
- Test various query types and edge cases
- Test memory constraints and performance limits
- Test error recovery and graceful degradation
```

### 3. Performance Tests
**What to Ask For:**
```
Create performance and load testing:
- tests/test_rag_performance.py
- Memory usage tests (stay within 100MB limit)
- Response time tests (< 5 seconds for RAG queries)
- Concurrent user simulation
- Large document set testing (300 pages, 75 PDFs)
- Search performance benchmarks
- Memory leak detection
```

### 4. FIPS Compliance Tests
**What to Ask For:**
```
Create FIPS compliance validation:
- tests/test_fips_compliance.py
- Validate no MD5 usage throughout RAG system
- Test cryptographic operations in FIPS mode
- Verify hash algorithm compliance (SHA-256 only)
- Test with OPENSSL_FIPS=1 environment
- Validate document hashing and storage
```

### 5. Test Data and Fixtures
**What to Ask For:**
```
Create test data management:
- tests/fixtures/ directory with sample PDFs
- Mock document generators for testing
- Test configuration profiles
- Sample query sets for different scenarios
- Expected response templates
- Performance benchmark data
```

### 6. Demo Environment Validation
**What to Ask For:**
```
Create demo-specific validation tests:
- tests/test_demo_scenario.py
- Test with exactly 75 PDFs and 300 pages
- Validate memory usage under demo constraints
- Test upload workflow with realistic documents
- Validate search quality with demo content
- Test user interface components with demo data
```

### 7. CI/CD Test Integration
**What to Ask For:**
```
Set up automated testing:
- Update existing test runners to include RAG tests
- Create test configuration for CI/CD
- Add performance regression detection
- FIPS compliance checking in automated builds
- Test reporting and failure analysis
```

## Expected Outputs After This Chat
- [ ] Complete unit test suite for all RAG components
- [ ] Integration tests covering end-to-end workflows
- [ ] Performance tests validating memory and speed requirements
- [ ] FIPS compliance validation tests
- [ ] Test data fixtures and mock generators
- [ ] Demo environment validation suite
- [ ] Automated test integration and reporting

## Key Implementation Details

### Unit Test Examples

#### Document Processor Tests
```python
# tests/test_rag_document_processor.py
import pytest
from unittest.mock import Mock, patch
from rag.document_processor import DoclingProcessor

class TestDoclingProcessor:
    def test_process_pdf_basic(self):
        """Test basic PDF processing functionality"""
        processor = DoclingProcessor(mock_config)
        
        with patch('docling.DocumentConverter') as mock_converter:
            # Mock Docling response
            mock_result = Mock()
            mock_result.document.export_to_markdown.return_value = "Sample text"
            mock_converter.return_value.convert.return_value = mock_result
            
            result = processor.process_pdf("test.pdf")
            
            assert result is not None
            assert "text" in result
            assert "chunks" in result
            assert "metadata" in result
    
    def test_chunking_preserves_structure(self):
        """Test that chunking preserves document structure"""
        processor = DoclingProcessor(mock_config)
        
        # Test with structured text
        structured_text = """
        # Introduction
        This is the introduction section.
        
        ## Background
        This is background information.
        """
        
        chunks = processor.create_chunks(structured_text)
        
        # Verify structure preservation
        assert len(chunks) > 0
        assert any("Introduction" in chunk["text"] for chunk in chunks)
        assert all(chunk["char_start"] < chunk["char_end"] for chunk in chunks)
    
    def test_fips_compliant_hashing(self):
        """Test that document hashing uses FIPS-compliant algorithms"""
        processor = DoclingProcessor(mock_config)
        
        test_content = b"test document content"
        doc_hash = processor._generate_document_hash(test_content)
        
        # Verify SHA-256 hash format (64 hex characters)
        assert len(doc_hash) == 64
        assert all(c in "0123456789abcdef" for c in doc_hash)
```

#### File Storage Tests
```python
# tests/test_rag_file_storage.py
import pytest
import tempfile
import os
from rag.file_storage import FileStorage

class TestFileStorage:
    def test_store_and_retrieve_document(self):
        """Test document storage and retrieval"""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(temp_dir)
            
            # Mock document result
            doc_result = {
                "doc_id": "test123",
                "filename": "test.pdf",
                "chunks": [{"chunk_id": "chunk1", "text": "test content"}],
                "metadata": {"pages": 1}
            }
            
            # Store document
            doc_id = storage.store_document(doc_result)
            assert doc_id == "test123"
            
            # Retrieve document
            retrieved = storage.load_document(doc_id)
            assert retrieved is not None
            assert retrieved["filename"] == "test.pdf"
    
    def test_memory_management(self):
        """Test memory usage limits and LRU eviction"""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(temp_dir, max_memory_mb=1)  # Very low limit
            
            # Store multiple documents to trigger eviction
            for i in range(10):
                large_doc = {
                    "doc_id": f"doc{i}",
                    "filename": f"test{i}.pdf",
                    "chunks": [{"text": "x" * 10000}],  # Large content
                    "metadata": {"pages": 1}
                }
                storage.store_document(large_doc)
            
            # Verify memory management is working
            assert storage._get_memory_usage() <= storage.max_memory
```

#### Search Engine Tests
```python
# tests/test_rag_search_engine.py
import pytest
from rag.search_engine import TFIDFSearchEngine
from rag.file_storage import FileStorage

class TestTFIDFSearchEngine:
    def test_build_search_index(self):
        """Test search index building"""
        mock_storage = Mock()
        mock_storage.get_all_chunks.return_value = [
            {"chunk_id": "1", "text": "machine learning algorithms"},
            {"chunk_id": "2", "text": "deep neural networks"},
            {"chunk_id": "3", "text": "natural language processing"}
        ]
        
        search_engine = TFIDFSearchEngine(mock_storage)
        search_engine.build_index()
        
        assert search_engine.tfidf_matrix is not None
        assert len(search_engine.chunk_metadata) == 3
    
    def test_similarity_search(self):
        """Test similarity search functionality"""
        search_engine = self._create_test_search_engine()
        
        results = search_engine.search_similar_chunks(
            "machine learning", 
            top_k=2
        )
        
        assert len(results) <= 2
        assert all(result.similarity_score >= 0 for result in results)
        assert results[0].similarity_score >= results[1].similarity_score
    
    def test_relevance_threshold_filtering(self):
        """Test that low-relevance results are filtered out"""
        search_engine = self._create_test_search_engine()
        
        # Search for something unrelated
        results = search_engine.search_similar_chunks(
            "cooking recipes",
            top_k=5,
            relevance_threshold=0.5
        )
        
        # Should return fewer results due to filtering
        assert all(result.similarity_score >= 0.5 for result in results)
```

### Integration Tests

#### End-to-End Workflow Test
```python
# tests/test_rag_integration.py
import pytest
import tempfile
from rag.rag_pipeline import RAGPipeline
from rag.document_processor import DoclingProcessor
from rag.file_storage import FileStorage
from rag.search_engine import TFIDFSearchEngine

class TestRAGIntegration:
    def test_complete_rag_workflow(self):
        """Test complete workflow from document to response"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up components
            storage = FileStorage(temp_dir)
            search_engine = TFIDFSearchEngine(storage)
            mock_ai_client = Mock()
            mock_ai_client.chat.return_value = "Based on the context, the answer is..."
            
            pipeline = RAGPipeline(storage, search_engine, mock_ai_client)
            
            # Process a test document
            test_pdf_path = "tests/fixtures/sample.pdf"
            processor = DoclingProcessor(test_config)
            doc_result = processor.process_pdf(test_pdf_path)
            storage.store_document(doc_result)
            
            # Build search index
            search_engine.build_index()
            
            # Test RAG query
            response = pipeline.process_rag_query("What is machine learning?")
            
            assert response.response is not None
            assert len(response.sources) > 0
            assert response.context_used is not None
    
    def test_multiple_documents_scenario(self):
        """Test RAG with multiple documents"""
        # Test uploading multiple PDFs and querying across them
        pass
    
    def test_memory_constraints_integration(self):
        """Test system behavior under memory constraints"""
        # Test with memory limits and large document sets
        pass
```

### Performance Tests

#### Memory and Speed Validation
```python
# tests/test_rag_performance.py
import pytest
import time
import psutil
import os
from rag.rag_pipeline import RAGPipeline

class TestRAGPerformance:
    def test_memory_usage_limits(self):
        """Test that RAG system stays within memory limits"""
        initial_memory = psutil.Process().memory_info().rss
        
        # Process target document set (300 pages, 75 PDFs)
        pipeline = self._setup_demo_pipeline()
        
        # Load and process all demo documents
        for pdf_file in self._get_demo_pdfs():
            pipeline.process_document(pdf_file)
        
        current_memory = psutil.Process().memory_info().rss
        memory_increase = current_memory - initial_memory
        
        # Should stay under 100MB increase
        assert memory_increase < 100 * 1024 * 1024
    
    def test_query_response_time(self):
        """Test that RAG queries respond within time limits"""
        pipeline = self._setup_test_pipeline_with_data()
        
        test_queries = [
            "What is the main topic?",
            "Explain the methodology",
            "What are the conclusions?"
        ]
        
        for query in test_queries:
            start_time = time.time()
            response = pipeline.process_rag_query(query)
            response_time = time.time() - start_time
            
            # Should respond within 5 seconds
            assert response_time < 5.0
            assert response.response is not None
    
    def test_concurrent_users(self):
        """Test system performance with multiple concurrent users"""
        import threading
        
        pipeline = self._setup_test_pipeline_with_data()
        results = {}
        
        def user_session(user_id):
            start_time = time.time()
            response = pipeline.process_rag_query(f"Query from user {user_id}")
            results[user_id] = time.time() - start_time
        
        # Simulate 5 concurrent users
        threads = []
        for i in range(5):
            thread = threading.Thread(target=user_session, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All users should get responses within reasonable time
        assert all(time < 10.0 for time in results.values())
```

### FIPS Compliance Tests

```python
# tests/test_fips_compliance.py
import pytest
import os
import subprocess
from rag.document_processor import DoclingProcessor
from rag.file_storage import FileStorage

class TestFIPSCompliance:
    def test_no_md5_usage(self):
        """Test that no MD5 algorithms are used"""
        # Set FIPS mode
        os.environ["OPENSSL_FIPS"] = "1"
        
        try:
            # Test document processing
            processor = DoclingProcessor(fips_config)
            test_content = b"test document for FIPS validation"
            doc_hash = processor._generate_document_hash(test_content)
            
            # Should not raise FIPS violation
            assert len(doc_hash) == 64  # SHA-256 hash length
            
        finally:
            # Clean up environment
            if "OPENSSL_FIPS" in os.environ:
                del os.environ["OPENSSL_FIPS"]
    
    def test_storage_operations_fips_compliant(self):
        """Test that all storage operations are FIPS compliant"""
        os.environ["OPENSSL_FIPS"] = "1"
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                storage = FileStorage(temp_dir)
                
                # Test document storage
                test_doc = {
                    "doc_id": "fips_test",
                    "filename": "test.pdf",
                    "chunks": [{"text": "test content"}],
                    "metadata": {"pages": 1}
                }
                
                # Should not raise FIPS violations
                doc_id = storage.store_document(test_doc)
                retrieved = storage.load_document(doc_id)
                
                assert retrieved is not None
                
        finally:
            if "OPENSSL_FIPS" in os.environ:
                del os.environ["OPENSSL_FIPS"]
```

## Test Data Management

### Sample PDF Generation
```python
# tests/fixtures/generate_test_pdfs.py
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def generate_sample_pdf(filename: str, content: str):
    """Generate a sample PDF for testing"""
    c = canvas.Canvas(filename, pagesize=letter)
    
    # Add content to PDF
    text_lines = content.split('\n')
    y_position = 750
    
    for line in text_lines:
        c.drawString(100, y_position, line)
        y_position -= 20
        
        if y_position < 100:  # New page
            c.showPage()
            y_position = 750
    
    c.save()

# Generate test PDFs for demo scenario
def create_demo_test_data():
    """Create test data that matches demo requirements"""
    # Generate 75 PDFs with total of 300 pages
    pages_per_doc = [4, 4, 4, 4, 4] * 15  # 4 pages each for 75 docs
    
    for i, page_count in enumerate(pages_per_doc):
        content = f"""
        Document {i+1}
        
        This is a test document for RAG validation.
        
        Topic: Test Topic {(i % 10) + 1}
        
        Content section with relevant information about machine learning,
        artificial intelligence, and document processing.
        
        This document has {page_count} pages total.
        """
        
        # Extend content to fill pages
        extended_content = content * (page_count * 5)
        generate_sample_pdf(f"tests/fixtures/demo_doc_{i+1:02d}.pdf", extended_content)
```

## Success Criteria for This Chat
- ✅ Comprehensive unit tests cover all RAG components
- ✅ Integration tests validate end-to-end workflows
- ✅ Performance tests confirm memory and speed requirements
- ✅ FIPS compliance tests pass in FIPS-enabled environment
- ✅ Test data supports realistic demo scenarios
- ✅ All tests can be run automatically in CI/CD
- ✅ Test coverage > 90% for RAG components

## Next Chat Guide
After completing testing and validation, proceed to **Chat Guide 08: OpenShift Deployment** to create the final deployment configuration and documentation.

## Notes for Implementation
- Use pytest for consistent testing framework
- Mock external dependencies (Docling, AI clients) for unit tests
- Create realistic test data that matches demo requirements
- Test both success and failure scenarios
- Include performance regression detection
- Ensure tests work in FIPS-enabled environments
- Document test setup and execution procedures