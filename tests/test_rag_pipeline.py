"""
Tests for RAG Pipeline Integration.
Tests end-to-end RAG functionality, context management, and quality controls.
"""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

from rag.rag_pipeline import (
    RAGPipeline, ContextManager, RAGQualityController, RAGPromptTemplates,
    SourceCitation, RAGResponse
)
from rag.file_storage import FileStorage
from rag.search_engine import TFIDFSearchEngine, SearchResult
from rag.document_processor import DocumentResult, DocumentMetadata, DocumentChunk, ChunkMetadata


class TestSourceCitation(unittest.TestCase):
    """Test SourceCitation data structure."""
    
    def test_source_citation_creation(self):
        """Test creating and serializing source citations."""
        citation = SourceCitation(
            document="test.pdf",
            page_number=1,
            section_title="Introduction",
            excerpt="This is a test excerpt",
            relevance_score=0.8,
            chunk_id="chunk_001",
            doc_id="doc_123"
        )
        
        self.assertEqual(citation.document, "test.pdf")
        self.assertEqual(citation.page_number, 1)
        self.assertEqual(citation.relevance_score, 0.8)
        
        # Test serialization
        citation_dict = citation.to_dict()
        self.assertIn('document', citation_dict)
        self.assertIn('relevance_score', citation_dict)
        self.assertEqual(citation_dict['page_number'], 1)


class TestRAGResponse(unittest.TestCase):
    """Test RAGResponse data structure."""
    
    def test_rag_response_creation(self):
        """Test creating and serializing RAG responses."""
        sources = [
            SourceCitation(
                document="test.pdf",
                page_number=1,
                section_title="Introduction",
                excerpt="Test excerpt",
                relevance_score=0.8,
                chunk_id="chunk_001",
                doc_id="doc_123"
            )
        ]
        
        response = RAGResponse(
            response="This is a test response",
            sources=sources,
            context_used="Test context",
            relevance_scores=[0.8, 0.6],
            processing_time=1.5,
            rag_mode="full",
            query="What is the test about?"
        )
        
        self.assertEqual(response.response, "This is a test response")
        self.assertEqual(response.rag_mode, "full")
        self.assertEqual(len(response.sources), 1)
        
        # Test serialization
        response_dict = response.to_dict()
        self.assertIn('response', response_dict)
        self.assertIn('sources', response_dict)
        self.assertIn('timestamp', response_dict)


class TestContextManager(unittest.TestCase):
    """Test context management and token limits."""
    
    def setUp(self):
        self.context_manager = ContextManager(max_tokens=1000)
    
    def test_token_estimation(self):
        """Test token count estimation."""
        text = "This is a test string with several words"
        tokens = self.context_manager.estimate_tokens(text)
        
        # Should be roughly len(text) / 4
        expected_tokens = len(text) // 4
        self.assertAlmostEqual(tokens, expected_tokens, delta=5)
    
    def test_context_selection_empty(self):
        """Test context selection with empty results."""
        context, sources = self.context_manager.select_best_context([], "test query")
        
        self.assertEqual(context, "")
        self.assertEqual(len(sources), 0)
    
    def test_context_selection_within_limits(self):
        """Test context selection within token limits."""
        # Create mock search results
        search_results = [
            SearchResult(
                chunk_id="chunk_001",
                doc_id="doc_123",
                text="This is the first chunk of text content",
                similarity_score=0.9,
                page_number=1,
                section_title="Introduction",
                filename="test.pdf",
                snippet="first chunk",
                context_before="",
                context_after="",
                combined_score=0.9
            ),
            SearchResult(
                chunk_id="chunk_002",
                doc_id="doc_123",
                text="This is the second chunk of text content",
                similarity_score=0.8,
                page_number=1,
                section_title="Introduction",
                filename="test.pdf",
                snippet="second chunk",
                context_before="",
                context_after="",
                combined_score=0.8
            )
        ]
        
        context, sources = self.context_manager.select_best_context(
            search_results, "test query"
        )
        
        self.assertTrue(len(context) > 0)
        self.assertEqual(len(sources), 2)  # Both chunks should fit
        self.assertIn("first chunk", context)
        self.assertIn("second chunk", context)
    
    def test_context_selection_exceeds_limits(self):
        """Test context selection when content exceeds token limits."""
        # Create a very long text that exceeds limits
        long_text = "This is a very long text. " * 200  # Make it very long
        
        search_results = [
            SearchResult(
                chunk_id="chunk_001",
                doc_id="doc_123",
                text=long_text,
                similarity_score=0.9,
                page_number=1,
                section_title="Introduction",
                filename="test.pdf",
                snippet="long text",
                context_before="",
                context_after="",
                combined_score=0.9
            )
        ]
        
        context, sources = self.context_manager.select_best_context(
            search_results, "test query"
        )
        
        # Should still return something, but truncated
        self.assertTrue(len(context) > 0)
        self.assertEqual(len(sources), 1)
        # Context should be within reasonable limits
        estimated_tokens = self.context_manager.estimate_tokens(context)
        self.assertLess(estimated_tokens, self.context_manager.max_tokens)
    
    def test_merge_overlapping_chunks(self):
        """Test merging of overlapping chunks from same document/page."""
        search_results = [
            SearchResult(
                chunk_id="chunk_001",
                doc_id="doc_123",
                text="First chunk",
                similarity_score=0.9,
                page_number=1,
                section_title="Introduction",
                filename="test.pdf",
                snippet="first",
                context_before="",
                context_after="",
                combined_score=0.9
            ),
            SearchResult(
                chunk_id="chunk_002",
                doc_id="doc_123",
                text="Second chunk",
                similarity_score=0.8,
                page_number=1,  # Same page
                section_title="Introduction",
                filename="test.pdf",
                snippet="second",
                context_before="",
                context_after="",
                combined_score=0.8
            ),
            SearchResult(
                chunk_id="chunk_003",
                doc_id="doc_456",
                text="Third chunk",
                similarity_score=0.7,
                page_number=2,  # Different doc
                section_title="Methods",
                filename="test2.pdf",
                snippet="third",
                context_before="",
                context_after="",
                combined_score=0.7
            )
        ]
        
        merged = self.context_manager.merge_overlapping_chunks(search_results)
        
        # Should merge chunks from same doc/page, keep the higher scoring one
        self.assertEqual(len(merged), 2)  # doc_123_1 and doc_456_2
        
        # The first result should be the higher scoring one from doc_123 page 1
        first_result = next(r for r in merged if r.doc_id == "doc_123")
        self.assertEqual(first_result.combined_score, 0.9)


class TestRAGQualityController(unittest.TestCase):
    """Test RAG quality controls and safeguards."""
    
    def setUp(self):
        self.quality_controller = RAGQualityController(
            min_relevance=0.1,
            min_context_length=50
        )
    
    def test_context_quality_assessment(self):
        """Test context quality assessment."""
        query = "machine learning algorithms"
        
        # High quality context
        good_context = "Machine learning algorithms are computational methods that learn patterns from data. These algorithms include supervised learning methods like neural networks and decision trees."
        
        quality_score = self.quality_controller.assess_context_quality(good_context, query)
        self.assertGreater(quality_score, 0.5)
        
        # Poor quality context
        poor_context = "This is unrelated content about cooking recipes."
        
        quality_score = self.quality_controller.assess_context_quality(poor_context, query)
        self.assertLess(quality_score, 0.3)
        
        # Empty context
        quality_score = self.quality_controller.assess_context_quality("", query)
        self.assertEqual(quality_score, 0.0)
    
    def test_should_use_rag_decision(self):
        """Test RAG usage decision based on search results."""
        # Good results - should use RAG
        good_results = [
            SearchResult(
                chunk_id="chunk_001",
                doc_id="doc_123",
                text="This is relevant content with good length and relevance",
                similarity_score=0.8,
                page_number=1,
                section_title="Introduction",
                filename="test.pdf",
                snippet="relevant content",
                context_before="",
                context_after="",
                combined_score=0.8
            )
        ]
        
        should_use, reason = self.quality_controller.should_use_rag(good_results)
        self.assertTrue(should_use)
        self.assertEqual(reason, "Quality checks passed")
        
        # Poor results - should not use RAG
        poor_results = [
            SearchResult(
                chunk_id="chunk_001",
                doc_id="doc_123",
                text="Short",  # Too short
                similarity_score=0.05,  # Too low relevance
                page_number=1,
                section_title="Introduction",
                filename="test.pdf",
                snippet="short",
                context_before="",
                context_after="",
                combined_score=0.05
            )
        ]
        
        should_use, reason = self.quality_controller.should_use_rag(poor_results)
        self.assertFalse(should_use)
        self.assertIn("relevance threshold", reason)
        
        # Empty results
        should_use, reason = self.quality_controller.should_use_rag([])
        self.assertFalse(should_use)
        self.assertEqual(reason, "No search results found")
    
    def test_hallucination_detection(self):
        """Test basic hallucination detection."""
        context = "The study involved 100 participants and took place over 6 months."
        
        # Response that contradicts context
        response_with_hallucination = "The study involved exactly 500 participants and was conducted over exactly 12 months."
        
        is_hallucination = self.quality_controller.detect_potential_hallucination(
            response_with_hallucination, context
        )
        self.assertTrue(is_hallucination)
        
        # Response consistent with context
        good_response = "Based on the study with 100 participants conducted over 6 months..."
        
        is_hallucination = self.quality_controller.detect_potential_hallucination(
            good_response, context
        )
        self.assertFalse(is_hallucination)


class TestRAGPromptTemplates(unittest.TestCase):
    """Test RAG prompt templates and formatting."""
    
    def test_format_sources_list(self):
        """Test formatting of sources list."""
        sources = [
            SourceCitation(
                document="test1.pdf",
                page_number=1,
                section_title="Introduction",
                excerpt="First excerpt",
                relevance_score=0.8,
                chunk_id="chunk_001",
                doc_id="doc_123"
            ),
            SourceCitation(
                document="test2.pdf",
                page_number=5,
                section_title="Methods",
                excerpt="Second excerpt",
                relevance_score=0.7,
                chunk_id="chunk_002",
                doc_id="doc_456"
            )
        ]
        
        sources_text = RAGPromptTemplates.format_sources_list(sources)
        
        self.assertIn("Source 1: test1.pdf, Page 1", sources_text)
        self.assertIn("Source 2: test2.pdf, Page 5", sources_text)
        self.assertIn("Introduction", sources_text)
        self.assertIn("Methods", sources_text)
    
    def test_get_prompt_for_mode(self):
        """Test getting appropriate prompts for different RAG modes."""
        query = "What is machine learning?"
        context = "Machine learning is a subset of AI"
        sources = [
            SourceCitation(
                document="ml_book.pdf",
                page_number=1,
                section_title="Introduction",
                excerpt="ML definition",
                relevance_score=0.9,
                chunk_id="chunk_001",
                doc_id="doc_123"
            )
        ]
        
        # Test full RAG prompt
        full_prompt = RAGPromptTemplates.get_prompt_for_mode(
            "full", query, context, sources
        )
        self.assertIn(query, full_prompt)
        self.assertIn(context, full_prompt)
        self.assertIn("ml_book.pdf", full_prompt)
        
        # Test partial RAG prompt
        partial_prompt = RAGPromptTemplates.get_prompt_for_mode(
            "partial", query, context, sources
        )
        self.assertIn("may not fully answer", partial_prompt)
        
        # Test fallback prompt
        fallback_prompt = RAGPromptTemplates.get_prompt_for_mode(
            "fallback", query, "", []
        )
        self.assertIn("No relevant context", fallback_prompt)
        
        # Test invalid mode
        with self.assertRaises(ValueError):
            RAGPromptTemplates.get_prompt_for_mode(
                "invalid_mode", query, context, sources
            )


class TestRAGPipeline(unittest.TestCase):
    """Test complete RAG pipeline integration."""
    
    def setUp(self):
        """Set up test environment with mocked components."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock storage
        self.mock_storage = Mock(spec=FileStorage)
        
        # Mock search engine
        self.mock_search_engine = Mock(spec=TFIDFSearchEngine)
        
        # Mock AI client
        self.mock_ai_client = Mock()
        self.mock_ai_client.chat.return_value = iter(["This is a test response"])
        
        # Create RAG pipeline
        self.rag_pipeline = RAGPipeline(
            storage=self.mock_storage,
            search_engine=self.mock_search_engine,
            ai_client=self.mock_ai_client,
            max_context_tokens=1000
        )
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_rag_pipeline_initialization(self):
        """Test RAG pipeline initialization."""
        self.assertIsNotNone(self.rag_pipeline.context_manager)
        self.assertIsNotNone(self.rag_pipeline.quality_controller)
        self.assertIsNotNone(self.rag_pipeline.prompt_templates)
        self.assertEqual(self.rag_pipeline.stats['total_queries'], 0)
    
    def test_full_rag_query_processing(self):
        """Test complete RAG query processing in full mode."""
        # Setup mock search results
        mock_search_results = [
            SearchResult(
                chunk_id="chunk_001",
                doc_id="doc_123",
                text="Machine learning is a subset of artificial intelligence that focuses on algorithms",
                similarity_score=0.9,
                page_number=1,
                section_title="Introduction to ML",
                filename="ml_guide.pdf",
                snippet="Machine learning is a subset",
                context_before="",
                context_after="",
                combined_score=0.9
            )
        ]
        
        self.mock_search_engine.search_similar_chunks.return_value = mock_search_results
        
        # Process query
        query = "What is machine learning?"
        model = "test-model"
        
        response = self.rag_pipeline.process_rag_query(
            query=query,
            model=model,
            max_sources=5,
            relevance_threshold=0.1
        )
        
        # Verify response
        self.assertIsInstance(response, RAGResponse)
        self.assertEqual(response.query, query)
        self.assertEqual(response.rag_mode, "full")
        self.assertGreater(len(response.sources), 0)
        self.assertGreater(response.processing_time, 0)
        
        # Verify search engine was called
        self.mock_search_engine.search_similar_chunks.assert_called_once()
        
        # Verify AI client was called
        self.mock_ai_client.chat.assert_called_once()
        
        # Verify stats were updated
        self.assertEqual(self.rag_pipeline.stats['total_queries'], 1)
        self.assertEqual(self.rag_pipeline.stats['full_rag_queries'], 1)
    
    def test_fallback_mode_processing(self):
        """Test RAG processing when falling back to non-RAG mode."""
        # Setup mock to return poor search results
        mock_poor_results = [
            SearchResult(
                chunk_id="chunk_001",
                doc_id="doc_123",
                text="Short",  # Too short and irrelevant
                similarity_score=0.01,  # Too low relevance
                page_number=1,
                section_title="Irrelevant",
                filename="test.pdf",
                snippet="Short",
                context_before="",
                context_after="",
                combined_score=0.01
            )
        ]
        
        self.mock_search_engine.search_similar_chunks.return_value = mock_poor_results
        
        # Process query
        response = self.rag_pipeline.process_rag_query(
            query="What is quantum computing?",
            model="test-model"
        )
        
        # Should fallback to non-RAG mode
        self.assertEqual(response.rag_mode, "fallback")
        self.assertEqual(len(response.sources), 0)
        self.assertEqual(response.context_used, "")
        
        # Verify stats
        self.assertEqual(self.rag_pipeline.stats['fallback_queries'], 1)
    
    def test_error_handling(self):
        """Test error handling in RAG pipeline."""
        # Make search engine raise an exception
        self.mock_search_engine.search_similar_chunks.side_effect = Exception("Search failed")
        
        # Process query - should handle error gracefully
        response = self.rag_pipeline.process_rag_query(
            query="Test query",
            model="test-model"
        )
        
        # Should return fallback response
        self.assertEqual(response.rag_mode, "fallback")
        self.assertIsInstance(response, RAGResponse)
    
    def test_pipeline_stats(self):
        """Test pipeline statistics tracking."""
        # Process multiple queries
        self.mock_search_engine.search_similar_chunks.return_value = []
        
        for i in range(3):
            self.rag_pipeline.process_rag_query(
                query=f"Test query {i}",
                model="test-model"
            )
        
        stats = self.rag_pipeline.get_pipeline_stats()
        
        self.assertEqual(stats['total_queries'], 3)
        self.assertEqual(stats['fallback_queries'], 3)  # All should be fallback due to empty results
        self.assertGreater(stats['avg_processing_time'], 0)
        self.assertEqual(stats['fallback_rate'], 1.0)
    
    def test_generate_response_with_context(self):
        """Test response generation with context."""
        query = "What is AI?"
        context = "Artificial Intelligence is the simulation of human intelligence"
        sources = [
            SourceCitation(
                document="ai_book.pdf",
                page_number=1,
                section_title="Introduction",
                excerpt="AI definition",
                relevance_score=0.9,
                chunk_id="chunk_001",
                doc_id="doc_123"
            )
        ]
        
        # Test response generation
        response_text = self.rag_pipeline.generate_response_with_context(
            query=query,
            context=context,
            sources=sources,
            rag_mode="full",
            model="test-model"
        )
        
        # Should return the mocked response
        self.assertEqual(response_text, "This is a test response")
        
        # Verify AI client was called with proper messages
        self.mock_ai_client.chat.assert_called()
        call_args = self.mock_ai_client.chat.call_args
        
        # Check that context and sources were included in the prompt
        messages = call_args[1]['messages']
        prompt_content = messages[0]['content']
        self.assertIn(context, prompt_content)
        self.assertIn("ai_book.pdf", prompt_content)


if __name__ == '__main__':
    unittest.main() 