"""
RAG Pipeline Integration - Complete end-to-end RAG system.
Integrates search engine with AI chat interface for context-enhanced responses.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Generator
import threading

from .file_storage import FileStorage
from .search_engine import TFIDFSearchEngine, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class SourceCitation:
    """Source citation for RAG responses."""
    document: str
    page_number: int
    section_title: str
    excerpt: str
    relevance_score: float
    chunk_id: str
    doc_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'document': self.document,
            'page_number': self.page_number,
            'section_title': self.section_title,
            'excerpt': self.excerpt,
            'relevance_score': self.relevance_score,
            'chunk_id': self.chunk_id,
            'doc_id': self.doc_id
        }


@dataclass
class RAGResponse:
    """Complete RAG response with sources and metadata."""
    response: str
    sources: List[SourceCitation]
    context_used: str
    relevance_scores: List[float]
    processing_time: float
    rag_mode: str  # "full", "partial", "fallback"
    query: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'response': self.response,
            'sources': [source.to_dict() for source in self.sources],
            'context_used': self.context_used,
            'relevance_scores': self.relevance_scores,
            'processing_time': self.processing_time,
            'rag_mode': self.rag_mode,
            'query': self.query,
            'timestamp': self.timestamp
        }


class ContextManager:
    """Manages context selection and token limits for RAG."""
    
    def __init__(self, max_tokens: int = 3000):
        self.max_tokens = max_tokens
        self.token_safety_margin = 200  # Safety margin for prompt overhead
        
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: 4 chars per token)."""
        return len(text) // 4
    
    def select_best_context(
        self,
        search_results: List[SearchResult],
        query: str
    ) -> Tuple[str, List[SourceCitation]]:
        """Select best context within token limits."""
        if not search_results:
            return "", []
        
        available_tokens = self.max_tokens - self.token_safety_margin
        selected_chunks = []
        used_tokens = 0
        sources = []
        
        # Sort by combined score (best first)
        sorted_results = sorted(search_results, 
                              key=lambda x: x.combined_score, 
                              reverse=True)
        
        # Merge overlapping chunks first
        merged_results = self.merge_overlapping_chunks(sorted_results)
        
        for result in merged_results:
            chunk_tokens = self.estimate_tokens(result.text)
            
            if used_tokens + chunk_tokens <= available_tokens:
                selected_chunks.append(result.text)
                used_tokens += chunk_tokens
                
                # Create source citation
                source = SourceCitation(
                    document=result.filename,
                    page_number=result.page_number,
                    section_title=result.section_title,
                    excerpt=result.snippet,
                    relevance_score=result.combined_score,
                    chunk_id=result.chunk_id,
                    doc_id=result.doc_id
                )
                sources.append(source)
            else:
                # Try to fit a truncated version
                remaining_tokens = available_tokens - used_tokens
                if remaining_tokens > 100:  # Only if meaningful space left
                    truncated_text = self._truncate_to_tokens(result.text, remaining_tokens)
                    selected_chunks.append(truncated_text + "...")
                    
                    source = SourceCitation(
                        document=result.filename,
                        page_number=result.page_number,
                        section_title=result.section_title,
                        excerpt=result.snippet,
                        relevance_score=result.combined_score,
                        chunk_id=result.chunk_id,
                        doc_id=result.doc_id
                    )
                    sources.append(source)
                break
        
        # Build context string
        context = "\n\n".join(f"[Source {i+1}]: {chunk}" 
                             for i, chunk in enumerate(selected_chunks))
        
        return context, sources
    
    def merge_overlapping_chunks(self, search_results: List[SearchResult]) -> List[SearchResult]:
        """Merge chunks from same document/page to reduce redundancy."""
        merged = {}
        
        for result in search_results:
            key = f"{result.doc_id}_{result.page_number}"
            
            if key not in merged:
                merged[key] = result
            else:
                # Keep the higher-scoring chunk, but could enhance to merge text
                if result.combined_score > merged[key].combined_score:
                    merged[key] = result
        
        return list(merged.values())
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximately max_tokens."""
        target_chars = max_tokens * 4  # Rough approximation
        if len(text) <= target_chars:
            return text
        
        # Try to break at sentence boundary
        truncated = text[:target_chars]
        sentence_end = truncated.rfind('.')
        if sentence_end > target_chars * 0.7:  # If we found a reasonable break point
            return truncated[:sentence_end + 1]
        
        return truncated


class RAGQualityController:
    """Quality controls and safeguards for RAG responses."""
    
    def __init__(self, min_relevance: float = 0.1, min_context_length: int = 50):
        self.min_relevance = min_relevance
        self.min_context_length = min_context_length
    
    def assess_context_quality(self, context: str, query: str) -> float:
        """Assess quality of retrieved context."""
        if not context or len(context) < self.min_context_length:
            return 0.0
        
        # Simple quality metrics
        query_terms = set(query.lower().split())
        context_terms = set(context.lower().split())
        
        # Term overlap score
        overlap = len(query_terms.intersection(context_terms))
        overlap_score = min(overlap / len(query_terms), 1.0) if query_terms else 0.0
        
        # Length score (prefer substantial context)
        length_score = min(len(context) / 500, 1.0)  # Normalize to 500 chars
        
        # Combined quality score
        quality_score = (overlap_score * 0.7) + (length_score * 0.3)
        return quality_score
    
    def should_use_rag(self, search_results: List[SearchResult]) -> Tuple[bool, str]:
        """Determine if RAG should be used based on result quality."""
        if not search_results:
            return False, "No search results found"
        
        # Check if any results meet minimum relevance threshold
        high_quality_results = [r for r in search_results if r.combined_score >= self.min_relevance]
        
        if not high_quality_results:
            return False, f"No results above relevance threshold ({self.min_relevance})"
        
        # Check for sufficient context
        total_text_length = sum(len(r.text) for r in high_quality_results[:3])  # Top 3 results
        
        if total_text_length < self.min_context_length:
            return False, "Insufficient context length"
        
        return True, "Quality checks passed"
    
    def detect_potential_hallucination(self, response: str, context: str) -> bool:
        """Basic hallucination detection (can be enhanced)."""
        if not context:
            return False  # Can't detect without context
        
        # Check for specific factual claims that might not be in context
        # This is a basic implementation - could be enhanced with NLP techniques
        
        # Look for absolute statements that might be hallucinated
        hallucination_patterns = [
            r'\b(exactly|precisely|definitely|absolutely)\s+\d+',
            r'\bthe\s+only\s+way\b',
            r'\balways\s+true\b',
            r'\bnever\s+possible\b'
        ]
        
        for pattern in hallucination_patterns:
            if re.search(pattern, response.lower()):
                # Check if this claim is supported by context
                if not re.search(pattern, context.lower()):
                    logger.warning(f"Potential hallucination detected: {pattern}")
                    return True
        
        return False


class RAGPromptTemplates:
    """RAG-specific prompt templates and engineering."""
    
    RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context from documents. 

Guidelines:
1. Use the provided context to inform your response
2. Always cite your sources using [Source N] format when referencing specific information
3. If the context doesn't fully answer the question, you may supplement with general knowledge but clearly distinguish this
4. If the context contradicts your general knowledge, defer to the context
5. Be precise and factual, avoiding speculation

Context:
{context}

Available Sources:
{sources}

Question: {query}

Please provide a comprehensive answer and cite relevant sources."""
    
    FALLBACK_PROMPT = """The user asked: {query}

No relevant context was found in the available documents for this question.

Please provide a helpful response based on your general knowledge and clearly indicate that this response is not based on the provided documents. If appropriate, suggest what type of information or documents might be helpful for answering this question."""
    
    PARTIAL_RAG_PROMPT = """You are answering the following question: {query}

Some relevant context was found in the available documents, but it may not fully answer the question:

Context:
{context}

Available Sources:
{sources}

Please provide a comprehensive answer that:
1. Uses the available context where relevant, citing sources with [Source N] format
2. Supplements with your general knowledge where the context is insufficient
3. Clearly distinguishes between information from the provided context and your general knowledge
4. Indicates any limitations in the available information"""
    
    @classmethod
    def format_sources_list(cls, sources: List[SourceCitation]) -> str:
        """Format sources list for prompt inclusion."""
        if not sources:
            return "No sources available."
        
        sources_text = []
        for i, source in enumerate(sources, 1):
            sources_text.append(
                f"Source {i}: {source.document}, Page {source.page_number}"
                f" (Section: {source.section_title})"
            )
        
        return "\n".join(sources_text)
    
    @classmethod
    def get_prompt_for_mode(cls, mode: str, query: str, context: str, sources: List[SourceCitation]) -> str:
        """Get appropriate prompt based on RAG mode."""
        sources_text = cls.format_sources_list(sources)
        
        if mode == "full":
            return cls.RAG_SYSTEM_PROMPT.format(
                context=context,
                sources=sources_text,
                query=query
            )
        elif mode == "partial":
            return cls.PARTIAL_RAG_PROMPT.format(
                query=query,
                context=context,
                sources=sources_text
            )
        elif mode == "fallback":
            return cls.FALLBACK_PROMPT.format(query=query)
        else:
            raise ValueError(f"Unknown RAG mode: {mode}")


class RAGPipeline:
    """Complete RAG pipeline integrating search, context management, and generation."""
    
    def __init__(
        self,
        storage: FileStorage,
        search_engine: TFIDFSearchEngine,
        ai_client,  # UnifiedAPIClient
        max_context_tokens: int = 3000,
        quality_controller: Optional[RAGQualityController] = None
    ):
        self.storage = storage
        self.search_engine = search_engine
        self.ai_client = ai_client
        
        # Initialize components
        self.context_manager = ContextManager(max_context_tokens)
        self.quality_controller = quality_controller or RAGQualityController()
        self.prompt_templates = RAGPromptTemplates()
        
        # Stats tracking
        self.stats = {
            'total_queries': 0,
            'full_rag_queries': 0,
            'partial_rag_queries': 0,
            'fallback_queries': 0,
            'total_processing_time': 0.0
        }
        
        self.lock = threading.RLock()
        
        logger.info("RAG Pipeline initialized")
    
    def process_rag_query(
        self,
        query: str,
        model: str,
        max_sources: int = 5,
        relevance_threshold: float = 0.1,
        stream: bool = True,
        **kwargs
    ) -> RAGResponse:
        """Process query through complete RAG pipeline."""
        start_time = time.time()
        
        with self.lock:
            self.stats['total_queries'] += 1
        
        try:
            # Step 1: Search for relevant context
            logger.info(f"Processing RAG query: {query[:100]}...")
            search_results = self.search_engine.search_similar_chunks(
                query=query,
                top_k=max_sources * 2,  # Get extra results for better selection
                filters={'min_score': relevance_threshold}
            )
            
            # Step 2: Quality assessment and mode determination
            should_use_rag, quality_reason = self.quality_controller.should_use_rag(search_results)
            
            if should_use_rag:
                # Filter to best results
                filtered_results = [r for r in search_results[:max_sources] 
                                  if r.combined_score >= relevance_threshold]
                
                # Build context
                context, sources = self.context_manager.select_best_context(
                    filtered_results, query
                )
                
                # Assess context quality
                context_quality = self.quality_controller.assess_context_quality(context, query)
                
                # Determine RAG mode
                if context_quality > 0.7:
                    rag_mode = "full"
                    with self.lock:
                        self.stats['full_rag_queries'] += 1
                elif context_quality > 0.3:
                    rag_mode = "partial"
                    with self.lock:
                        self.stats['partial_rag_queries'] += 1
                else:
                    rag_mode = "fallback"
                    context = ""
                    sources = []
                    with self.lock:
                        self.stats['fallback_queries'] += 1
            else:
                # Use fallback mode
                logger.info(f"Using fallback mode: {quality_reason}")
                rag_mode = "fallback"
                context = ""
                sources = []
                filtered_results = []
                with self.lock:
                    self.stats['fallback_queries'] += 1
            
            # Step 3: Generate response
            response_text = self.generate_response_with_context(
                query=query,
                context=context,
                sources=sources,
                rag_mode=rag_mode,
                model=model,
                stream=stream,
                **kwargs
            )
            
            # Step 4: Quality check response
            if context and self.quality_controller.detect_potential_hallucination(response_text, context):
                logger.warning("Potential hallucination detected in response")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            with self.lock:
                self.stats['total_processing_time'] += processing_time
            
            # Build response object
            relevance_scores = [r.combined_score for r in filtered_results]
            
            return RAGResponse(
                response=response_text,
                sources=sources,
                context_used=context,
                relevance_scores=relevance_scores,
                processing_time=processing_time,
                rag_mode=rag_mode,
                query=query
            )
            
        except Exception as e:
            logger.error(f"Error processing RAG query: {e}")
            # Return fallback response on error
            processing_time = time.time() - start_time
            
            fallback_response = self.generate_response_with_context(
                query=query,
                context="",
                sources=[],
                rag_mode="fallback",
                model=model,
                stream=stream,
                **kwargs
            )
            
            return RAGResponse(
                response=fallback_response,
                sources=[],
                context_used="",
                relevance_scores=[],
                processing_time=processing_time,
                rag_mode="fallback",
                query=query
            )
    
    def generate_response_with_context(
        self,
        query: str,
        context: str,
        sources: List[SourceCitation],
        rag_mode: str,
        model: str,
        stream: bool = True,
        **kwargs
    ) -> str:
        """Generate AI response with context and citations."""
        try:
            # Get appropriate prompt
            prompt = self.prompt_templates.get_prompt_for_mode(
                mode=rag_mode,
                query=query,
                context=context,
                sources=sources
            )
            
            # Prepare messages for AI client
            messages = [{"role": "user", "content": prompt}]
            
            # Generate response
            response_parts = []
            for chunk in self.ai_client.chat(
                model=model,
                messages=messages,
                stream=stream,
                **kwargs
            ):
                response_parts.append(chunk)
            
            return "".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        with self.lock:
            stats = self.stats.copy()
        
        # Calculate derived stats
        if stats['total_queries'] > 0:
            stats['avg_processing_time'] = stats['total_processing_time'] / stats['total_queries']
            stats['full_rag_rate'] = stats['full_rag_queries'] / stats['total_queries']
            stats['partial_rag_rate'] = stats['partial_rag_queries'] / stats['total_queries'] 
            stats['fallback_rate'] = stats['fallback_queries'] / stats['total_queries']
        else:
            stats['avg_processing_time'] = 0.0
            stats['full_rag_rate'] = 0.0
            stats['partial_rag_rate'] = 0.0
            stats['fallback_rate'] = 0.0
        
        return stats
    
    def clear_stats(self):
        """Clear pipeline statistics."""
        with self.lock:
            self.stats = {
                'total_queries': 0,
                'full_rag_queries': 0,
                'partial_rag_queries': 0,
                'fallback_queries': 0,
                'total_processing_time': 0.0
            } 