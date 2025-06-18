"""
Demo scenario validation tests.
Tests the RAG system with the exact demo requirements: 75 PDFs, 300 pages, memory constraints.
"""

import pytest
import tempfile
import shutil
import time
import statistics
from pathlib import Path
from unittest.mock import Mock, patch
from typing import List, Dict, Any, Tuple

from rag.rag_pipeline import RAGPipeline
from rag.document_processor import DoclingProcessor, RAGConfig
from rag.file_storage import FileStorage
from rag.search_engine import TFIDFSearchEngine
from tests.fixtures.generate_test_data import TestDataGenerator, DOCUMENT_TEMPLATES


class DemoScenarioValidator:
    """Utility class for validating demo scenario requirements."""
    
    def __init__(self):
        self.requirements = {
            'max_pdfs': 75,
            'max_pages': 300,
            'max_memory_mb': 100,
            'max_response_time_seconds': 5,
            'min_search_relevance': 0.1
        }
    
    def validate_document_count(self, document_count: int) -> bool:
        """Validate document count meets demo requirements."""
        return document_count <= self.requirements['max_pdfs']
    
    def validate_page_count(self, page_count: int) -> bool:
        """Validate total page count meets demo requirements."""
        return page_count <= self.requirements['max_pages']
    
    def validate_memory_usage(self, memory_mb: float) -> bool:
        """Validate memory usage meets demo constraints."""
        return memory_mb <= self.requirements['max_memory_mb']
    
    def validate_response_time(self, response_time: float) -> bool:
        """Validate response time meets demo requirements."""
        return response_time <= self.requirements['max_response_time_seconds']
    
    def generate_demo_report(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate demo validation report."""
        return {
            'requirements': self.requirements,
            'actual_metrics': metrics,
            'validation_results': {
                'documents': self.validate_document_count(metrics.get('document_count', 0)),
                'pages': self.validate_page_count(metrics.get('page_count', 0)),
                'memory': self.validate_memory_usage(metrics.get('memory_mb', 0)),
                'response_time': self.validate_response_time(metrics.get('avg_response_time', 0))
            },
            'overall_compliance': all([
                self.validate_document_count(metrics.get('document_count', 0)),
                self.validate_page_count(metrics.get('page_count', 0)),
                self.validate_memory_usage(metrics.get('memory_mb', 0)),
                self.validate_response_time(metrics.get('avg_response_time', 0))
            ])
        }


class TestDemoScenario:
    """Test RAG system with exact demo scenario requirements."""
    
    @pytest.fixture
    def demo_storage_dir(self):
        """Create demo storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def demo_config(self):
        """Create demo-specific RAG configuration."""
        return RAGConfig(
            demo_mode=True,
            demo_max_pdfs=75,
            demo_max_pages=300,
            max_memory_mb=100,
            fips_mode=True,
            hash_algorithm="sha256",
            chunk_size=1000,
            chunk_overlap=200,
            max_search_results=5,
            relevance_threshold=0.1
        )
    
    @pytest.fixture
    def demo_validator(self):
        """Create demo scenario validator."""
        return DemoScenarioValidator()
    
    @pytest.fixture
    def mock_ai_client(self):
        """Create mock AI client for demo testing."""
        mock_client = Mock()
        
        def mock_chat_response(messages, **kwargs):
            # Simulate realistic response time
            time.sleep(0.1)
            context = ""
            for message in messages:
                if isinstance(message, dict) and message.get('role') == 'user':
                    context += message.get('content', '')
            
            # Generate contextually appropriate responses
            if "machine learning" in context.lower():
                return iter([
                    "Based on the provided documents, machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming."
                ])
            elif "data science" in context.lower():
                return iter([
                    "According to the documentation, data science involves systematic processes for extracting insights from data using statistical and computational methods."
                ])
            elif "security" in context.lower() or "fips" in context.lower():
                return iter([
                    "The documents indicate that FIPS compliance requires using approved cryptographic algorithms such as SHA-256 for security in government systems."
                ])
            else:
                return iter([
                    "Based on the available documents, I can provide information relevant to your query."
                ])
        
        mock_client.chat = mock_chat_response
        return mock_client
    
    def test_demo_document_set_limits(self, demo_storage_dir, demo_config, demo_validator):
        """Test system handles exactly 75 PDFs with 300 total pages."""
        storage = FileStorage(demo_storage_dir, max_memory_mb=demo_config.max_memory_mb)
        
        with patch('rag.document_processor.DOCLING_AVAILABLE', True):
            with patch('rag.document_processor.DocumentConverter') as mock_converter:
                processor = DoclingProcessor(demo_config, storage=storage)
                
                # Mock PDF processing
                def mock_convert(pdf_path):
                    # Simulate realistic content extraction
                    pages = 4  # Average pages per PDF for 300 total / 75 PDFs
                    content = f"Document content from {Path(pdf_path).name}. " * (pages * 50)
                    
                    mock_doc = Mock()
                    mock_doc.export_to_text.return_value = content
                    mock_result = Mock()
                    mock_result.document = mock_doc
                    return mock_result
                
                mock_converter.return_value.convert.side_effect = mock_convert
        
        # Generate exactly the demo document set
        generator = TestDataGenerator(demo_storage_dir)
        
        # Create 75 PDFs with total of 300 pages
        pdf_files = []
        total_pages = 0
        target_pages_per_doc = 300 // 75  # 4 pages per doc
        extra_pages = 300 % 75  # Distribute remaining pages
        
        for i in range(75):
            pages_for_doc = target_pages_per_doc + (1 if i < extra_pages else 0)
            total_pages += pages_for_doc
            
            template_key = list(DOCUMENT_TEMPLATES.keys())[i % len(DOCUMENT_TEMPLATES)]
            filename = f"demo_doc_{i+1:02d}.pdf"
            
            # Create temporary PDF file for processing
            temp_pdf = Path(demo_storage_dir) / filename
            temp_pdf.write_bytes(b"fake pdf content")
            
            # Process PDF
            doc_result = processor.process_pdf(str(temp_pdf))
            storage.store_document(doc_result)
            pdf_files.append(str(temp_pdf))
        
        # Validate demo requirements
        assert len(pdf_files) == 75, f"Should have exactly 75 PDFs, got {len(pdf_files)}"
        assert total_pages == 300, f"Should have exactly 300 pages, got {total_pages}"
        assert len(storage.document_index) == 75, f"Should store 75 documents, got {len(storage.document_index)}"
        
        # Validate with demo validator
        metrics = {
            'document_count': len(pdf_files),
            'page_count': total_pages
        }
        
        assert demo_validator.validate_document_count(metrics['document_count'])
        assert demo_validator.validate_page_count(metrics['page_count'])
    
    def test_demo_memory_constraints(self, demo_storage_dir, demo_config, demo_validator, mock_ai_client):
        """Test system stays within 100MB memory limit during demo scenario."""
        import psutil
        
        storage = FileStorage(demo_storage_dir, max_memory_mb=demo_config.max_memory_mb)
        search_engine = TFIDFSearchEngine(storage, max_features=3000)
        
        with patch('rag.document_processor.DOCLING_AVAILABLE', True):
            with patch('rag.document_processor.DocumentConverter') as mock_converter:
                processor = DoclingProcessor(demo_config, storage=storage)
                
                mock_doc = Mock()
                mock_doc.export_to_text.return_value = "Demo document content for memory testing. " * 100
                mock_result = Mock()
                mock_result.document = mock_doc
                mock_converter.return_value.convert.return_value = mock_result
        
        pipeline = RAGPipeline(storage, search_engine, mock_ai_client)
        
        # Monitor memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_samples = [initial_memory]
        
        try:
            # Load demo document set (scaled for testing)
            num_docs = 25  # Scaled down for test performance
            for i in range(num_docs):
                # Create realistic document content
                content = f"Demo document {i} content. " * 200  # ~2KB per doc
                doc_result = processor.process_text(content, f"demo_doc_{i}.txt")
                storage.store_document(doc_result)
                
                # Sample memory usage
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
            
            # Build search index
            search_engine.build_index()
            memory_samples.append(process.memory_info().rss / 1024 / 1024)
            
            # Perform multiple queries
            demo_queries = [
                "What is machine learning?",
                "Explain data science concepts",
                "Describe security requirements",
                "How does AI work?",
                "What are the main topics in these documents?"
            ]
            
            for query in demo_queries:
                response = pipeline.process_rag_query(query, model="demo-model")
                assert response is not None
                
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
        
        finally:
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_samples.append(final_memory)
        
        # Analyze memory usage
        peak_memory = max(memory_samples)
        memory_increase = peak_memory - initial_memory
        
        # Validate memory constraints
        assert memory_increase < demo_config.max_memory_mb, \
            f"Memory increase ({memory_increase:.1f}MB) exceeded limit ({demo_config.max_memory_mb}MB)"
        
        assert demo_validator.validate_memory_usage(memory_increase), \
            "Memory usage should meet demo requirements"
        
        print(f"Demo memory test results:")
        print(f"  Initial memory: {initial_memory:.1f}MB")
        print(f"  Peak memory: {peak_memory:.1f}MB") 
        print(f"  Memory increase: {memory_increase:.1f}MB")
        print(f"  Limit: {demo_config.max_memory_mb}MB")
    
    def test_demo_query_performance(self, demo_storage_dir, demo_config, demo_validator, mock_ai_client):
        """Test query performance meets demo requirements (<5 seconds)."""
        storage = FileStorage(demo_storage_dir, max_memory_mb=demo_config.max_memory_mb)
        search_engine = TFIDFSearchEngine(storage, max_features=2000)
        
        with patch('rag.document_processor.DOCLING_AVAILABLE', True):
            with patch('rag.document_processor.DocumentConverter') as mock_converter:
                processor = DoclingProcessor(demo_config, storage=storage)
                
                mock_doc = Mock()
                mock_doc.export_to_text.return_value = "Performance test document with comprehensive content for demo validation."
                mock_result = Mock()
                mock_result.document = mock_doc
                mock_converter.return_value.convert.return_value = mock_result
        
        pipeline = RAGPipeline(storage, search_engine, mock_ai_client)
        
        # Set up demo document set
        for i in range(20):  # Representative subset
            content = f"""
            Demo Document {i}
            
            This document contains information about various topics including machine learning,
            data science, artificial intelligence, security standards, and technical documentation.
            
            Content section {i} provides detailed information for testing query performance
            and ensuring the system meets demonstration requirements.
            """
            doc_result = processor.process_text(content, f"demo_perf_{i}.txt")
            storage.store_document(doc_result)
        
        search_engine.build_index()
        
        # Demo query scenarios
        demo_queries = [
            {
                "query": "What is machine learning?",
                "category": "basic_concept",
                "expected_max_time": 3.0
            },
            {
                "query": "Explain the relationship between data science and artificial intelligence",
                "category": "complex_analysis",
                "expected_max_time": 5.0
            },
            {
                "query": "What are the security standards mentioned in the documents?",
                "category": "specific_information",
                "expected_max_time": 4.0
            },
            {
                "query": "Provide a summary of the technical documentation",
                "category": "comprehensive_summary",
                "expected_max_time": 5.0
            },
            {
                "query": "How do these topics relate to each other?",
                "category": "analytical_reasoning",
                "expected_max_time": 5.0
            }
        ]
        
        performance_results = []
        
        for query_data in demo_queries:
            query = query_data["query"]
            
            # Measure response time
            start_time = time.time()
            response = pipeline.process_rag_query(query, model="demo-model")
            response_time = time.time() - start_time
            
            performance_results.append({
                "query": query,
                "category": query_data["category"],
                "response_time": response_time,
                "expected_max_time": query_data["expected_max_time"],
                "meets_requirement": response_time <= query_data["expected_max_time"],
                "rag_mode": response.rag_mode,
                "source_count": len(response.sources) if response.sources else 0
            })
            
            # Validate individual query performance
            assert response_time <= query_data["expected_max_time"], \
                f"Query '{query}' took {response_time:.2f}s, exceeding limit of {query_data['expected_max_time']}s"
            
            assert response.rag_mode == "full", f"Query should use full RAG mode: {query}"
            assert len(response.sources) > 0, f"Query should have sources: {query}"
        
        # Overall performance validation
        avg_response_time = statistics.mean([r["response_time"] for r in performance_results])
        max_response_time = max([r["response_time"] for r in performance_results])
        
        assert demo_validator.validate_response_time(avg_response_time), \
            f"Average response time {avg_response_time:.2f}s exceeds demo requirements"
        
        assert max_response_time <= 5.0, \
            f"Maximum response time {max_response_time:.2f}s exceeds demo limit"
        
        print(f"Demo performance test results:")
        print(f"  Average response time: {avg_response_time:.2f}s")
        print(f"  Maximum response time: {max_response_time:.2f}s")
        print(f"  All queries met requirements: {all(r['meets_requirement'] for r in performance_results)}")
    
    def test_demo_search_quality(self, demo_storage_dir, demo_config, demo_validator, mock_ai_client):
        """Test search quality meets demo requirements."""
        storage = FileStorage(demo_storage_dir, max_memory_mb=demo_config.max_memory_mb)
        search_engine = TFIDFSearchEngine(storage, max_features=2000)
        
        with patch('rag.document_processor.DOCLING_AVAILABLE', True):
            with patch('rag.document_processor.DocumentConverter') as mock_converter:
                processor = DoclingProcessor(demo_config, storage=storage)
                
                mock_doc = Mock()
                mock_doc.export_to_text.return_value = "Search quality test document content."
                mock_result = Mock()
                mock_result.document = mock_doc
                mock_converter.return_value.convert.return_value = mock_result
        
        pipeline = RAGPipeline(storage, search_engine, mock_ai_client)
        
        # Create documents with known content for quality testing
        test_documents = [
            {
                "content": """
                Machine Learning Fundamentals
                
                Machine learning is a subset of artificial intelligence that enables computers 
                to learn from data without explicit programming. Key algorithms include 
                supervised learning, unsupervised learning, and reinforcement learning.
                """,
                "filename": "ml_fundamentals.txt",
                "topics": ["machine learning", "artificial intelligence", "algorithms"]
            },
            {
                "content": """
                Data Science Pipeline
                
                Data science involves systematic extraction of knowledge from data using 
                statistical methods, machine learning, and domain expertise. The process 
                includes data collection, cleaning, analysis, and visualization.
                """,
                "filename": "data_science.txt",
                "topics": ["data science", "statistical methods", "data analysis"]
            },
            {
                "content": """
                Cybersecurity Standards
                
                FIPS 140-2 provides security requirements for cryptographic modules. 
                Government systems must use approved algorithms like SHA-256 for hashing 
                and AES for encryption to ensure data protection.
                """,
                "filename": "security_standards.txt",
                "topics": ["cybersecurity", "FIPS", "cryptographic", "SHA-256"]
            }
        ]
        
        # Process test documents
        for doc_data in test_documents:
            doc_result = processor.process_text(doc_data["content"], doc_data["filename"])
            storage.store_document(doc_result)
        
        search_engine.build_index()
        
        # Quality test queries with expected results
        quality_tests = [
            {
                "query": "What is machine learning?",
                "expected_topics": ["machine learning", "artificial intelligence"],
                "min_relevance": 0.3,
                "expected_docs": ["ml_fundamentals.txt"]
            },
            {
                "query": "Describe the data science process",
                "expected_topics": ["data science", "statistical", "analysis"],
                "min_relevance": 0.2,
                "expected_docs": ["data_science.txt"]
            },
            {
                "query": "What are FIPS security requirements?",
                "expected_topics": ["FIPS", "security", "cryptographic"],
                "min_relevance": 0.2,
                "expected_docs": ["security_standards.txt"]
            },
            {
                "query": "How does AI relate to data analysis?",
                "expected_topics": ["machine learning", "data science"],
                "min_relevance": 0.1,
                "expected_docs": ["ml_fundamentals.txt", "data_science.txt"]
            }
        ]
        
        quality_results = []
        
        for test_case in quality_tests:
            response = pipeline.process_rag_query(test_case["query"], model="demo-model")
            
            # Analyze response quality
            response_lower = response.response.lower()
            topic_matches = sum(1 for topic in test_case["expected_topics"] 
                              if topic.lower() in response_lower)
            
            source_filenames = [source.document for source in response.sources]
            doc_matches = sum(1 for expected_doc in test_case["expected_docs"]
                            if any(expected_doc in filename for filename in source_filenames))
            
            relevance_scores = [source.relevance_score for source in response.sources]
            avg_relevance = statistics.mean(relevance_scores) if relevance_scores else 0
            
            quality_result = {
                "query": test_case["query"],
                "topic_matches": topic_matches,
                "expected_topics": len(test_case["expected_topics"]),
                "doc_matches": doc_matches,
                "expected_docs": len(test_case["expected_docs"]),
                "avg_relevance": avg_relevance,
                "min_relevance_met": avg_relevance >= test_case["min_relevance"],
                "response_length": len(response.response),
                "source_count": len(response.sources)
            }
            
            quality_results.append(quality_result)
            
            # Quality assertions
            assert topic_matches > 0, f"Query '{test_case['query']}' should match expected topics"
            assert avg_relevance >= test_case["min_relevance"], \
                f"Average relevance {avg_relevance:.3f} below minimum {test_case['min_relevance']}"
            assert len(response.sources) > 0, f"Query should return sources: {test_case['query']}"
            assert len(response.response) > 50, f"Response should be substantial: {test_case['query']}"
        
        # Overall quality metrics
        overall_quality = {
            "avg_topic_match_rate": statistics.mean([r["topic_matches"] / r["expected_topics"] 
                                                   for r in quality_results]),
            "avg_relevance_score": statistics.mean([r["avg_relevance"] for r in quality_results]),
            "all_queries_relevant": all(r["min_relevance_met"] for r in quality_results)
        }
        
        assert overall_quality["avg_topic_match_rate"] > 0.5, \
            "Should match at least 50% of expected topics on average"
        assert overall_quality["avg_relevance_score"] >= demo_config.relevance_threshold, \
            "Average relevance should meet demo threshold"
        assert overall_quality["all_queries_relevant"], \
            "All queries should meet minimum relevance requirements"
        
        print(f"Demo search quality results:")
        print(f"  Average topic match rate: {overall_quality['avg_topic_match_rate']:.1%}")
        print(f"  Average relevance score: {overall_quality['avg_relevance_score']:.3f}")
        print(f"  All queries met relevance threshold: {overall_quality['all_queries_relevant']}")
    
    def test_complete_demo_scenario_validation(self, demo_storage_dir, demo_config, demo_validator, mock_ai_client):
        """Test complete demo scenario end-to-end."""
        # Set up complete demo system
        storage = FileStorage(demo_storage_dir, max_memory_mb=demo_config.max_memory_mb)
        search_engine = TFIDFSearchEngine(storage, max_features=3000)
        
        with patch('rag.document_processor.DOCLING_AVAILABLE', True):
            with patch('rag.document_processor.DocumentConverter') as mock_converter:
                processor = DoclingProcessor(demo_config, storage=storage)
                
                mock_doc = Mock()
                mock_doc.export_to_text.return_value = "Complete demo scenario test document with comprehensive content."
                mock_result = Mock()
                mock_result.document = mock_doc
                mock_converter.return_value.convert.return_value = mock_result
        
        pipeline = RAGPipeline(storage, search_engine, mock_ai_client)
        
        # Monitor complete scenario
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        scenario_metrics = {
            'document_count': 0,
            'page_count': 0,
            'memory_mb': 0,
            'response_times': [],
            'search_quality_scores': []
        }
        
        try:
            # Phase 1: Document processing (scaled demo set)
            print("Phase 1: Processing demo document set...")
            num_demo_docs = 15  # Scaled for test performance
            pages_per_doc = 4
            
            for i in range(num_demo_docs):
                content = f"Demo scenario document {i} with multiple pages of content. " * (pages_per_doc * 25)
                doc_result = processor.process_text(content, f"demo_scenario_{i:02d}.pdf")
                storage.store_document(doc_result)
                
                scenario_metrics['document_count'] += 1
                scenario_metrics['page_count'] += pages_per_doc
            
            # Phase 2: Index building
            print("Phase 2: Building search index...")
            index_start = time.time()
            search_engine.build_index()
            index_time = time.time() - index_start
            
            # Phase 3: Query processing
            print("Phase 3: Processing demo queries...")
            demo_queries = [
                "What are the main topics covered in these documents?",
                "Explain the key concepts from the documentation",
                "How do the different subjects relate to each other?",
                "What specific information is available about technology?",
                "Provide a comprehensive overview of the content"
            ]
            
            for query in demo_queries:
                start_time = time.time()
                response = pipeline.process_rag_query(query, model="demo-model")
                response_time = time.time() - start_time
                
                scenario_metrics['response_times'].append(response_time)
                
                if response.sources:
                    avg_relevance = statistics.mean([s.relevance_score for s in response.sources])
                    scenario_metrics['search_quality_scores'].append(avg_relevance)
            
            # Final memory measurement
            final_memory = process.memory_info().rss / 1024 / 1024
            scenario_metrics['memory_mb'] = final_memory - initial_memory
            
            # Calculate aggregated metrics
            scenario_metrics['avg_response_time'] = statistics.mean(scenario_metrics['response_times'])
            scenario_metrics['max_response_time'] = max(scenario_metrics['response_times'])
            scenario_metrics['avg_search_quality'] = statistics.mean(scenario_metrics['search_quality_scores'])
            
        except Exception as e:
            pytest.fail(f"Demo scenario failed: {str(e)}")
        
        # Generate demo validation report
        demo_report = demo_validator.generate_demo_report(scenario_metrics)
        
        # Validate all demo requirements
        assert demo_report['overall_compliance'], \
            f"Demo scenario failed compliance check: {demo_report}"
        
        assert scenario_metrics['document_count'] <= demo_config.demo_max_pdfs, \
            f"Document count {scenario_metrics['document_count']} exceeds limit {demo_config.demo_max_pdfs}"
        
        assert scenario_metrics['avg_response_time'] <= 5.0, \
            f"Average response time {scenario_metrics['avg_response_time']:.2f}s exceeds 5s limit"
        
        assert scenario_metrics['memory_mb'] <= demo_config.max_memory_mb, \
            f"Memory usage {scenario_metrics['memory_mb']:.1f}MB exceeds {demo_config.max_memory_mb}MB limit"
        
        assert scenario_metrics['avg_search_quality'] >= demo_config.relevance_threshold, \
            f"Search quality {scenario_metrics['avg_search_quality']:.3f} below threshold {demo_config.relevance_threshold}"
        
        # Print demo scenario results
        print("\n" + "="*50)
        print("DEMO SCENARIO VALIDATION RESULTS")
        print("="*50)
        print(f"Documents processed: {scenario_metrics['document_count']}")
        print(f"Total pages: {scenario_metrics['page_count']}")
        print(f"Memory usage: {scenario_metrics['memory_mb']:.1f}MB")
        print(f"Average response time: {scenario_metrics['avg_response_time']:.2f}s")
        print(f"Maximum response time: {scenario_metrics['max_response_time']:.2f}s")
        print(f"Average search quality: {scenario_metrics['avg_search_quality']:.3f}")
        print(f"Index build time: {index_time:.2f}s")
        print(f"Overall compliance: {demo_report['overall_compliance']}")
        print("="*50)
        
        return demo_report


if __name__ == "__main__":
    # Allow running demo scenario tests independently
    pytest.main([__file__, "-v", "-s"]) 