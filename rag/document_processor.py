"""
Document processing module using Docling for PDF extraction and text chunking.
Maintains FIPS compliance by using SHA-256 for all hashing operations.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logging.warning("Docling not available. PDF processing will be disabled.")

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata for a document chunk."""
    chunk_id: str
    page_number: int
    section_title: str
    char_start: int
    char_end: int
    word_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentChunk:
    """A chunk of text from a document with metadata."""
    text: str
    metadata: ChunkMetadata
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'metadata': self.metadata.to_dict()
        }


@dataclass
class DocumentMetadata:
    """Metadata for a processed document."""
    doc_id: str
    filename: str
    upload_timestamp: str
    total_pages: int
    total_chunks: int
    processing_status: str
    file_size_bytes: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentResult:
    """Result of document processing."""
    metadata: DocumentMetadata
    chunks: List[DocumentChunk]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'metadata': self.metadata.to_dict(),
            'chunks': [chunk.to_dict() for chunk in self.chunks]
        }


class RAGConfig:
    """Configuration for RAG document processing."""
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 min_chunk_size: int = 100,
                 max_chunk_size: int = 2000):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size


class DoclingProcessor:
    """
    Document processor using Docling for PDF extraction and intelligent chunking.
    Maintains FIPS compliance by using SHA-256 for all hashing operations.
    """
    
    def __init__(self, config: Optional[RAGConfig] = None, storage=None, auto_store: bool = False):
        """Initialize the DoclingProcessor with configuration."""
        self.config = config or RAGConfig()
        self.storage = storage
        self.auto_store = auto_store
        
        if not DOCLING_AVAILABLE:
            raise ImportError("Docling is required but not available. Please install docling dependencies.")
        
        # Initialize Docling converter with basic options
        # Note: Complex format_options will be added after verifying API compatibility
        self.converter = DocumentConverter()
        
        logger.info("DoclingProcessor initialized with FIPS-compliant configuration")
    
    def _generate_fips_hash(self, content: str) -> str:
        """Generate FIPS-compliant SHA-256 hash for content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove control characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        return text.strip()
    
    def _extract_section_title(self, text: str, position: int = 0) -> str:
        """Extract section title from text around given position."""
        # Look for headings (lines that are short and might be titles)
        lines = text[:position + 100].split('\n')
        
        for line in reversed(lines[-5:]):  # Check last 5 lines
            line = line.strip()
            if line and len(line) < 100 and not line.endswith('.'):
                # Likely a heading
                return line
        
        return "Content"
    
    def _find_sentence_boundaries(self, text: str, start: int, end: int) -> Tuple[int, int]:
        """Find sentence boundaries to avoid breaking mid-sentence."""
        # Look for sentence endings before the end position
        sentence_endings = ['.', '!', '?', '\n\n']
        
        # Find the last sentence ending before the target end
        best_end = end
        for i in range(end - 50, end):
            if i < len(text) and text[i] in sentence_endings:
                # Make sure we're not in the middle of an abbreviation
                if text[i] == '.' and i + 1 < len(text) and text[i + 1].islower():
                    continue
                best_end = i + 1
                break
        
        # Find the first sentence beginning after the target start
        best_start = start
        for i in range(start, min(start + 50, len(text))):
            if text[i] in sentence_endings and i + 1 < len(text):
                best_start = i + 1
                break
        
        return max(0, best_start), min(len(text), best_end)
    
    def create_chunks(self, text: str, page_mapping: Dict[int, int], 
                     section_mapping: Dict[int, str]) -> List[DocumentChunk]:
        """
        Create intelligent chunks from document text with structure preservation.
        
        Args:
            text: Full document text
            page_mapping: Mapping from character position to page number
            section_mapping: Mapping from character position to section title
        
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        text_length = len(text)
        
        if text_length == 0:
            return chunks
        
        chunk_id = 1
        start = 0
        
        while start < text_length:
            # Calculate initial end position
            end = min(start + self.config.chunk_size, text_length)
            
            # Adjust boundaries to avoid breaking sentences
            if end < text_length:
                start, end = self._find_sentence_boundaries(text, start, end)
            
            # Extract chunk text
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) < self.config.min_chunk_size and end < text_length:
                # Chunk too small, extend it
                end = min(start + self.config.min_chunk_size + 100, text_length)
                chunk_text = text[start:end].strip()
            
            if chunk_text:  # Only create non-empty chunks
                # Get page number for this chunk (use start position)
                page_num = page_mapping.get(start, 1)
                
                # Get section title for this chunk
                section_title = section_mapping.get(start, self._extract_section_title(text, start))
                
                # Count words
                word_count = len(chunk_text.split())
                
                # Create chunk metadata
                metadata = ChunkMetadata(
                    chunk_id=f"chunk_{chunk_id:03d}",
                    page_number=page_num,
                    section_title=section_title,
                    char_start=start,
                    char_end=end,
                    word_count=word_count
                )
                
                # Create chunk
                chunk = DocumentChunk(text=chunk_text, metadata=metadata)
                chunks.append(chunk)
                
                chunk_id += 1
            
            # Move to next chunk with overlap
            start = max(start + 1, end - self.config.chunk_overlap)
            
            # Prevent infinite loops
            if start >= text_length:
                break
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks
    
    def process_pdf(self, file_path: str) -> DocumentResult:
        """
        Process a PDF file and extract text with intelligent chunking.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            DocumentResult with metadata and chunks
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a valid PDF
            Exception: For other processing errors
        """
        path_obj = Path(file_path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if path_obj.suffix.lower() != '.pdf':
            raise ValueError(f"File is not a PDF: {file_path}")
        
        try:
            logger.info(f"Starting PDF processing: {file_path}")
            
            # Convert document using Docling
            result = self.converter.convert(file_path)
            
            if not result or not result.document:
                raise ValueError("Failed to convert PDF - no document content extracted")
            
            doc = result.document
            
            # Extract full text
            full_text = doc.export_to_text()
            full_text = self._clean_text(full_text)
            
            if not full_text:
                raise ValueError("No text content extracted from PDF")
            
            # Create mappings for page numbers and sections
            # For now, we'll use simple mappings - this can be enhanced with Docling's structure info
            page_mapping = {}  # char_pos -> page_num
            section_mapping = {}  # char_pos -> section_title
            
            # Simple page mapping (this would be enhanced with actual Docling page info)
            estimated_pages = max(1, len(full_text) // 2000)  # Rough estimate
            chars_per_page = len(full_text) // estimated_pages
            for i in range(len(full_text)):
                page_mapping[i] = (i // chars_per_page) + 1
                section_mapping[i] = "Content"  # Would be enhanced with actual section detection
            
            # Create chunks
            chunks = self.create_chunks(full_text, page_mapping, section_mapping)
            
            # Generate document hash for unique ID
            doc_hash = self._generate_fips_hash(full_text)
            
            # Get file stats
            file_stats = path_obj.stat()
            
            # Create document metadata
            doc_metadata = DocumentMetadata(
                doc_id=doc_hash,
                filename=path_obj.name,
                upload_timestamp=datetime.now().isoformat(),
                total_pages=estimated_pages,
                total_chunks=len(chunks),
                processing_status="completed",
                file_size_bytes=file_stats.st_size
            )
            
            result = DocumentResult(metadata=doc_metadata, chunks=chunks)
            
            # Auto-store if enabled and storage is available
            if self.auto_store and self.storage:
                try:
                    stored_doc_id = self.storage.store_document(result)
                    logger.info(f"Auto-stored document with ID: {stored_doc_id}")
                except Exception as e:
                    logger.error(f"Failed to auto-store document: {e}")
            
            logger.info(f"Successfully processed PDF: {len(chunks)} chunks created")
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            raise
    
    def process_text(self, text: str, filename: str = "text_input") -> DocumentResult:
        """
        Process raw text input with chunking.
        
        Args:
            text: Raw text to process
            filename: Name to use for the document
            
        Returns:
            DocumentResult with metadata and chunks
        """
        try:
            logger.info(f"Processing text input: {filename}")
            
            # Clean text
            clean_text = self._clean_text(text)
            
            if not clean_text:
                raise ValueError("No text content provided")
            
            # Simple mappings for text input
            page_mapping = {i: 1 for i in range(len(clean_text))}  # All on page 1
            section_mapping = {i: "Text Content" for i in range(len(clean_text))}
            
            # Create chunks
            chunks = self.create_chunks(clean_text, page_mapping, section_mapping)
            
            # Generate document hash
            doc_hash = self._generate_fips_hash(clean_text)
            
            # Create metadata
            doc_metadata = DocumentMetadata(
                doc_id=doc_hash,
                filename=filename,
                upload_timestamp=datetime.now().isoformat(),
                total_pages=1,
                total_chunks=len(chunks),
                processing_status="completed",
                file_size_bytes=len(text.encode('utf-8'))
            )
            
            result = DocumentResult(metadata=doc_metadata, chunks=chunks)
            
            # Auto-store if enabled and storage is available
            if self.auto_store and self.storage:
                try:
                    stored_doc_id = self.storage.store_document(result)
                    logger.info(f"Auto-stored document with ID: {stored_doc_id}")
                except Exception as e:
                    logger.error(f"Failed to auto-store document: {e}")
            
            logger.info(f"Successfully processed text: {len(chunks)} chunks created")
            return result
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            raise
    
    def process_and_store_pdf(self, file_path: str) -> str:
        """Process a PDF and store it, returning the document ID."""
        if not self.storage:
            raise ValueError("Storage not configured. Initialize processor with storage to use this method.")
        
        result = self.process_pdf(file_path)
        doc_id = self.storage.store_document(result)
        logger.info(f"Processed and stored PDF: {file_path} -> {doc_id}")
        return doc_id
    
    def process_and_store_text(self, text: str, filename: str = "text_input") -> str:
        """Process text and store it, returning the document ID."""
        if not self.storage:
            raise ValueError("Storage not configured. Initialize processor with storage to use this method.")
        
        result = self.process_text(text, filename)
        doc_id = self.storage.store_document(result)
        logger.info(f"Processed and stored text: {filename} -> {doc_id}")
        return doc_id
    
    def check_document_exists(self, content: str) -> Optional[str]:
        """Check if document with this content already exists in storage."""
        if not self.storage:
            return None
        
        doc_hash = self._generate_fips_hash(content)
        doc_result = self.storage.load_document(doc_hash)
        return doc_hash if doc_result else None