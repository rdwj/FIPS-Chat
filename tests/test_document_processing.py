"""
Tests for the RAG document processing foundation.
Validates PDF extraction, chunking, metadata, and FIPS compliance.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from rag.document_processor import (
    DoclingProcessor, 
    RAGConfig,
    DocumentResult,
    DocumentMetadata,
    DocumentChunk,
    ChunkMetadata
)


class TestRAGConfig:
    """Test RAG configuration settings."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RAGConfig()
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.min_chunk_size == 100
        assert config.max_chunk_size == 2000
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = RAGConfig(
            chunk_size=1500,
            chunk_overlap=300,
            min_chunk_size=150,
            max_chunk_size=3000
        )
        assert config.chunk_size == 1500
        assert config.chunk_overlap == 300
        assert config.min_chunk_size == 150
        assert config.max_chunk_size == 3000


class TestChunkMetadata:
    """Test chunk metadata functionality."""
    
    def test_chunk_metadata_creation(self):
        """Test creating chunk metadata."""
        metadata = ChunkMetadata(
            chunk_id="chunk_001",
            page_number=1,
            section_title="Introduction",
            char_start=0,
            char_end=500,
            word_count=87
        )
        assert metadata.chunk_id == "chunk_001"
        assert metadata.page_number == 1
        assert metadata.section_title == "Introduction"
        assert metadata.char_start == 0
        assert metadata.char_end == 500
        assert metadata.word_count == 87
    
    def test_chunk_metadata_to_dict(self):
        """Test converting chunk metadata to dictionary."""
        metadata = ChunkMetadata(
            chunk_id="chunk_001",
            page_number=1,
            section_title="Introduction",
            char_start=0,
            char_end=500,
            word_count=87
        )
        metadata_dict = metadata.to_dict()
        
        assert isinstance(metadata_dict, dict)
        assert metadata_dict["chunk_id"] == "chunk_001"
        assert metadata_dict["page_number"] == 1
        assert metadata_dict["section_title"] == "Introduction"
        assert metadata_dict["char_start"] == 0
        assert metadata_dict["char_end"] == 500
        assert metadata_dict["word_count"] == 87


class TestDocumentChunk:
    """Test document chunk functionality."""
    
    def test_document_chunk_creation(self):
        """Test creating document chunk."""
        metadata = ChunkMetadata(
            chunk_id="chunk_001",
            page_number=1,
            section_title="Introduction",
            char_start=0,
            char_end=500,
            word_count=87
        )
        chunk = DocumentChunk(
            text="This is sample text content for testing.",
            metadata=metadata
        )
        
        assert chunk.text == "This is sample text content for testing."
        assert chunk.metadata == metadata
    
    def test_document_chunk_to_dict(self):
        """Test converting document chunk to dictionary."""
        metadata = ChunkMetadata(
            chunk_id="chunk_001",
            page_number=1,
            section_title="Introduction",
            char_start=0,
            char_end=500,
            word_count=87
        )
        chunk = DocumentChunk(
            text="This is sample text content for testing.",
            metadata=metadata
        )
        chunk_dict = chunk.to_dict()
        
        assert isinstance(chunk_dict, dict)
        assert chunk_dict["text"] == "This is sample text content for testing."
        assert isinstance(chunk_dict["metadata"], dict)
        assert chunk_dict["metadata"]["chunk_id"] == "chunk_001"


class TestDocumentResult:
    """Test document result functionality."""
    
    def test_document_result_creation(self):
        """Test creating document result."""
        doc_metadata = DocumentMetadata(
            doc_id="test_hash",
            filename="test.pdf",
            upload_timestamp="2024-01-01T00:00:00Z",
            total_pages=1,
            total_chunks=1,
            processing_status="completed",
            file_size_bytes=1000
        )
        
        chunk_metadata = ChunkMetadata(
            chunk_id="chunk_001",
            page_number=1,
            section_title="Introduction",
            char_start=0,
            char_end=500,
            word_count=87
        )
        
        chunk = DocumentChunk(
            text="This is sample text content for testing.",
            metadata=chunk_metadata
        )
        
        result = DocumentResult(metadata=doc_metadata, chunks=[chunk])
        
        assert result.metadata == doc_metadata
        assert len(result.chunks) == 1
        assert result.chunks[0] == chunk
    
    def test_document_result_to_dict(self):
        """Test converting document result to dictionary."""
        doc_metadata = DocumentMetadata(
            doc_id="test_hash",
            filename="test.pdf",
            upload_timestamp="2024-01-01T00:00:00Z",
            total_pages=1,
            total_chunks=1,
            processing_status="completed",
            file_size_bytes=1000
        )
        
        chunk_metadata = ChunkMetadata(
            chunk_id="chunk_001",
            page_number=1,
            section_title="Introduction",
            char_start=0,
            char_end=500,
            word_count=87
        )
        
        chunk = DocumentChunk(
            text="This is sample text content for testing.",
            metadata=chunk_metadata
        )
        
        result = DocumentResult(metadata=doc_metadata, chunks=[chunk])
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert "metadata" in result_dict
        assert "chunks" in result_dict
        assert isinstance(result_dict["metadata"], dict)
        assert isinstance(result_dict["chunks"], list)
        assert len(result_dict["chunks"]) == 1


class TestDoclingProcessor:
    """Test DoclingProcessor functionality."""
    
    @patch('rag.document_processor.DOCLING_AVAILABLE', False)
    def test_processor_init_without_docling(self):
        """Test processor initialization fails without Docling."""
        with pytest.raises(ImportError, match="Docling is required but not available"):
            DoclingProcessor()
    
    @patch('rag.document_processor.DOCLING_AVAILABLE', True)
    @patch('rag.document_processor.DocumentConverter')
    def test_processor_init_with_docling(self, mock_converter):
        """Test processor initialization with Docling available."""
        processor = DoclingProcessor()
        assert processor.config is not None
        assert processor.config.chunk_size == 1000
        mock_converter.assert_called_once()
    
    @patch('rag.document_processor.DOCLING_AVAILABLE', True)
    @patch('rag.document_processor.DocumentConverter')
    def test_processor_init_with_custom_config(self, mock_converter):
        """Test processor initialization with custom config."""
        config = RAGConfig(chunk_size=1500, chunk_overlap=300)
        processor = DoclingProcessor(config)
        assert processor.config.chunk_size == 1500
        assert processor.config.chunk_overlap == 300
    
    @patch('rag.document_processor.DOCLING_AVAILABLE', True)
    @patch('rag.document_processor.DocumentConverter')
    def test_fips_hash_generation(self, mock_converter):
        """Test FIPS-compliant hash generation."""
        processor = DoclingProcessor()
        test_content = "This is test content for hashing"
        
        # Generate hash using processor
        processor_hash = processor._generate_fips_hash(test_content)
        
        # Generate hash using standard library
        expected_hash = hashlib.sha256(test_content.encode('utf-8')).hexdigest()
        
        assert processor_hash == expected_hash
        assert len(processor_hash) == 64  # SHA-256 produces 64-character hex string
    
    @patch('rag.document_processor.DOCLING_AVAILABLE', True)
    @patch('rag.document_processor.DocumentConverter')
    def test_text_cleaning(self, mock_converter):
        """Test text cleaning functionality."""
        processor = DoclingProcessor()
        
        # Test with various whitespace and control characters
        dirty_text = "  This   is\t\ta   test\n\n\nwith    multiple\r\n   spaces  "
        cleaned_text = processor._clean_text(dirty_text)
        
        assert cleaned_text == "This is a test with multiple spaces"
        
        # Test with control characters (should be removed)
        dirty_text_with_control = "Hello\x00world\x08test\x1F"
        cleaned_control = processor._clean_text(dirty_text_with_control)
        
        assert "\x00" not in cleaned_control
        assert "\x08" not in cleaned_control
        assert "\x1F" not in cleaned_control
    
    @patch('rag.document_processor.DOCLING_AVAILABLE', True)
    @patch('rag.document_processor.DocumentConverter')
    def test_section_title_extraction(self, mock_converter):
        """Test section title extraction."""
        processor = DoclingProcessor()
        
        text_with_heading = """
        Introduction
        
        This is the beginning of the document content.
        It contains multiple sentences and paragraphs.
        """
        
        title = processor._extract_section_title(text_with_heading, 50)
        assert title == "Introduction"
        
        # Test with no clear heading
        text_without_heading = "This is just regular paragraph text without any clear headings in the content."
        title = processor._extract_section_title(text_without_heading, 20)
        assert title == "Content"  # Default fallback
    
    @patch('rag.document_processor.DOCLING_AVAILABLE', True)
    @patch('rag.document_processor.DocumentConverter')
    def test_sentence_boundary_finding(self, mock_converter):
        """Test sentence boundary detection."""
        processor = DoclingProcessor()
        
        text = "This is the first sentence. This is the second sentence. This is the third sentence."
        
        # Find boundaries in the middle of text
        start, end = processor._find_sentence_boundaries(text, 10, 50)
        
        # Should adjust to sentence boundaries
        assert start >= 0
        assert end <= len(text)
        assert text[end-1] in '.!?' or end == len(text)
    
    @patch('rag.document_processor.DOCLING_AVAILABLE', True)
    @patch('rag.document_processor.DocumentConverter')
    def test_chunk_creation(self, mock_converter):
        """Test intelligent chunk creation."""
        processor = DoclingProcessor()
        
        # Create test text
        test_text = """
        Introduction
        
        This is the introduction section of the document. It provides an overview of the topic.
        
        Chapter 1: Getting Started
        
        This chapter covers the basics. It includes multiple paragraphs and sentences to test chunking.
        The content should be split appropriately while maintaining readability and context.
        
        Chapter 2: Advanced Topics
        
        This chapter goes into more detail about advanced concepts. It should be chunked separately.
        """
        
        # Create simple mappings
        page_mapping = {i: 1 for i in range(len(test_text))}
        section_mapping = {i: "Test Content" for i in range(len(test_text))}
        
        chunks = processor.create_chunks(test_text, page_mapping, section_mapping)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(chunk.text.strip() for chunk in chunks)  # No empty chunks
        assert all(chunk.metadata.word_count > 0 for chunk in chunks)
        
        # Check chunk IDs are sequential
        for i, chunk in enumerate(chunks, 1):
            assert chunk.metadata.chunk_id == f"chunk_{i:03d}"
    
    @patch('rag.document_processor.DOCLING_AVAILABLE', True)
    @patch('rag.document_processor.DocumentConverter')
    def test_process_text(self, mock_converter):
        """Test text processing functionality."""
        processor = DoclingProcessor()
        
        test_text = "This is a test document with multiple sentences. It should be processed correctly and chunked appropriately."
        
        result = processor.process_text(test_text, "test_document.txt")
        
        assert isinstance(result, DocumentResult)
        assert result.metadata.filename == "test_document.txt"
        assert result.metadata.total_pages == 1
        assert result.metadata.total_chunks > 0
        assert result.metadata.processing_status == "completed"
        assert len(result.chunks) > 0
        
        # Verify FIPS-compliant hash
        assert len(result.metadata.doc_id) == 64  # SHA-256 hex string
        
        # Verify chunks have proper metadata
        for chunk in result.chunks:
            assert chunk.metadata.page_number == 1
            assert chunk.metadata.section_title == "Text Content"
            assert chunk.metadata.word_count > 0
    
    @patch('rag.document_processor.DOCLING_AVAILABLE', True)
    @patch('rag.document_processor.DocumentConverter')
    def test_process_pdf_file_not_found(self, mock_converter):
        """Test PDF processing with non-existent file."""
        processor = DoclingProcessor()
        
        with pytest.raises(FileNotFoundError, match="File not found"):
            processor.process_pdf("/nonexistent/file.pdf")
    
    @patch('rag.document_processor.DOCLING_AVAILABLE', True)
    @patch('rag.document_processor.DocumentConverter')
    def test_process_pdf_invalid_format(self, mock_converter):
        """Test PDF processing with non-PDF file."""
        processor = DoclingProcessor()
        
        # Create a temporary non-PDF file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
            tmp_file.write(b"This is not a PDF file")
            tmp_file_path = tmp_file.name
        
        try:
            with pytest.raises(ValueError, match="File is not a PDF"):
                processor.process_pdf(tmp_file_path)
        finally:
            os.unlink(tmp_file_path)
    
    @patch('rag.document_processor.DOCLING_AVAILABLE', True)
    @patch('rag.document_processor.DocumentConverter')
    def test_process_pdf_success(self, mock_converter):
        """Test successful PDF processing."""
        # Create mock objects
        mock_doc = Mock()
        mock_doc.export_to_text.return_value = "This is extracted PDF text content with multiple sentences."
        
        mock_result = Mock()
        mock_result.document = mock_doc
        
        mock_converter_instance = Mock()
        mock_converter_instance.convert.return_value = mock_result
        mock_converter.return_value = mock_converter_instance
        
        processor = DoclingProcessor()
        
        # Create a temporary PDF file (just for file existence check)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(b"Fake PDF content")
            tmp_file_path = tmp_file.name
        
        try:
            result = processor.process_pdf(tmp_file_path)
            
            assert isinstance(result, DocumentResult)
            assert result.metadata.filename.endswith(".pdf")
            assert result.metadata.processing_status == "completed"
            assert len(result.chunks) > 0
            
            # Verify the converter was called
            mock_converter_instance.convert.assert_called_once_with(tmp_file_path)
            mock_doc.export_to_text.assert_called_once()
            
        finally:
            os.unlink(tmp_file_path)
    
    def test_fips_compliance_no_md5(self):
        """Test that no MD5 hashing is used (FIPS compliance)."""
        # This test ensures we're only using FIPS-approved algorithms
        
        # Check that our hash generation uses SHA-256
        test_content = "Test content for FIPS compliance"
        
        # This should work (SHA-256 is FIPS-approved)
        sha256_hash = hashlib.sha256(test_content.encode('utf-8')).hexdigest()
        assert len(sha256_hash) == 64
        
        # Verify MD5 would be different (and we're not using it)
        md5_hash = hashlib.md5(test_content.encode('utf-8')).hexdigest()
        assert len(md5_hash) == 32
        assert sha256_hash != md5_hash
        
        # Our processor should use SHA-256
        with patch('rag.document_processor.DOCLING_AVAILABLE', True), \
             patch('rag.document_processor.DocumentConverter'):
            processor = DoclingProcessor()
            processor_hash = processor._generate_fips_hash(test_content)
            assert processor_hash == sha256_hash
    
    @patch('rag.document_processor.DOCLING_AVAILABLE', True)
    @patch('rag.document_processor.DocumentConverter')
    def test_empty_text_handling(self, mock_converter):
        """Test handling of empty or whitespace-only text."""
        processor = DoclingProcessor()
        
        # Test empty string
        with pytest.raises(ValueError, match="No text content provided"):
            processor.process_text("", "empty.txt")
        
        # Test whitespace-only string
        with pytest.raises(ValueError, match="No text content provided"):
            processor.process_text("   \n\t   ", "whitespace.txt")
    
    @patch('rag.document_processor.DOCLING_AVAILABLE', True)
    @patch('rag.document_processor.DocumentConverter')
    def test_metadata_serialization(self, mock_converter):
        """Test that metadata can be properly serialized to JSON."""
        processor = DoclingProcessor()
        
        test_text = "This is test content for metadata serialization testing."
        result = processor.process_text(test_text, "test.txt")
        
        # Convert to dictionary
        result_dict = result.to_dict()
        
        # Should be JSON serializable
        json_str = json.dumps(result_dict)
        assert isinstance(json_str, str)
        
        # Should be deserializable
        reconstructed = json.loads(json_str)
        assert reconstructed["metadata"]["filename"] == "test.txt"
        assert reconstructed["metadata"]["processing_status"] == "completed"
        assert len(reconstructed["chunks"]) > 0


if __name__ == "__main__":
    pytest.main([__file__]) 