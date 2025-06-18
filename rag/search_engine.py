"""
TF-IDF based search and retrieval engine for RAG documents.
Implements hybrid search with multiple relevance signals for enhanced document retrieval.
"""

import json
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import threading
import hashlib

from .file_storage import FileStorage, DocumentIndex
from .document_processor import DocumentResult, DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result of a search operation with relevance scoring."""
    chunk_id: str
    doc_id: str
    text: str
    similarity_score: float
    page_number: int
    section_title: str
    filename: str
    snippet: str  # Highlighted excerpt
    context_before: str  # Surrounding context
    context_after: str
    combined_score: float  # Hybrid relevance score
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchStats:
    """Statistics about search operations."""
    total_searches: int
    average_response_time_ms: float
    cache_hit_rate: float
    index_size_bytes: int
    total_documents_indexed: int
    last_index_update: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SearchCache:
    """LRU cache for search results."""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, Tuple[List[SearchResult], float]] = OrderedDict()
        self.lock = threading.RLock()
    
    def get(self, query_hash: str) -> Optional[List[SearchResult]]:
        """Get cached results if available and not expired."""
        with self.lock:
            if query_hash in self.cache:
                results, timestamp = self.cache[query_hash]
                if time.time() - timestamp < self.ttl_seconds:
                    # Move to end (most recently used)
                    self.cache.move_to_end(query_hash)
                    return results
                else:
                    # Expired, remove
                    del self.cache[query_hash]
            return None
    
    def put(self, query_hash: str, results: List[SearchResult]):
        """Cache search results."""
        with self.lock:
            # Remove oldest if at capacity
            while len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            
            self.cache[query_hash] = (results, time.time())
    
    def clear(self):
        """Clear all cached results."""
        with self.lock:
            self.cache.clear()


class TFIDFSearchEngine:
    """
    TF-IDF based search engine with hybrid relevance scoring.
    Implements memory-efficient search for large document collections.
    """
    
    def __init__(self, storage: FileStorage, 
                 max_features: int = 5000,
                 cache_size: int = 100,
                 index_path: Optional[str] = None):
        """
        Initialize the search engine.
        
        Args:
            storage: File storage system for documents
            max_features: Maximum number of TF-IDF features
            cache_size: Size of search result cache
            index_path: Path to save/load search indexes
        """
        self.storage = storage
        self.max_features = max_features
        self.index_path = Path(index_path) if index_path else storage.storage_path / "search_indexes"
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # TF-IDF components (will be initialized in build_index)
        self.vectorizer = None
        self.tfidf_matrix = None
        self.chunk_metadata: List[Dict[str, Any]] = []
        self.doc_chunks_map: Dict[str, List[int]] = {}  # doc_id -> chunk indices
        
        # Search cache and stats
        self.cache = SearchCache(cache_size)
        self.search_stats = {
            'total_searches': 0,
            'total_response_time': 0.0,
            'cache_hits': 0
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Try to load existing index
        self._load_index()
        
        logger.info(f"TFIDFSearchEngine initialized with {max_features} max features")
    
    def _generate_query_hash(self, query: str, top_k: int, filters: Dict[str, Any]) -> str:
        """Generate hash for query caching."""
        query_key = f"{query}|{top_k}|{json.dumps(filters, sort_keys=True)}"
        return hashlib.sha256(query_key.encode()).hexdigest()[:16]
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for indexing and search."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\(\)]', ' ', text)
        return text.strip()
    
    def build_index(self, force_rebuild: bool = False) -> bool:
        """
        Build TF-IDF index from all documents in storage.
        
        Args:
            force_rebuild: Force rebuilding even if index exists
            
        Returns:
            True if index was built successfully
        """
        try:
            with self.lock:
                logger.info("Building TF-IDF search index...")
                start_time = time.time()
                
                # Get all documents
                documents = self.storage.list_documents()
                if not documents:
                    logger.warning("No documents found for indexing")
                    return False
                
                # Collect all chunks
                all_texts = []
                self.chunk_metadata = []
                self.doc_chunks_map = {}
                
                for doc_index in documents:
                    doc_result = self.storage.load_document(doc_index.doc_id)
                    if not doc_result:
                        logger.warning(f"Could not load document {doc_index.doc_id}")
                        continue
                    
                    chunk_indices = []
                    for chunk in doc_result.chunks:
                        # Preprocess chunk text
                        processed_text = self._preprocess_text(chunk.text)
                        all_texts.append(processed_text)
                        
                        # Store metadata
                        chunk_meta = {
                            'chunk_id': chunk.metadata.chunk_id,
                            'doc_id': doc_result.metadata.doc_id,
                            'text': chunk.text,  # Keep original text
                            'page_number': chunk.metadata.page_number,
                            'section_title': chunk.metadata.section_title,
                            'filename': doc_result.metadata.filename,
                            'word_count': chunk.metadata.word_count,
                            'char_start': chunk.metadata.char_start,
                            'char_end': chunk.metadata.char_end,
                            'upload_date': doc_result.metadata.upload_timestamp
                        }
                        self.chunk_metadata.append(chunk_meta)
                        chunk_indices.append(len(self.chunk_metadata) - 1)
                    
                    self.doc_chunks_map[doc_index.doc_id] = chunk_indices
                
                if not all_texts:
                    logger.error("No text content found for indexing")
                    return False
                
                # Adjust min_df based on corpus size for small document sets
                min_df = max(1, min(2, len(all_texts) // 3))
                max_df = min(0.8, (len(all_texts) - 1) / len(all_texts)) if len(all_texts) > 1 else 0.9
                
                # Create vectorizer with adaptive parameters
                self.vectorizer = TfidfVectorizer(
                    max_features=self.max_features,
                    stop_words='english',
                    ngram_range=(1, 2),
                    lowercase=True,
                    token_pattern=r'\b[a-zA-Z]{2,}\b',
                    max_df=max_df,
                    min_df=min_df
                )
                
                # Build TF-IDF matrix
                logger.info(f"Fitting TF-IDF vectorizer on {len(all_texts)} chunks...")
                self.tfidf_matrix = self.vectorizer.fit_transform(all_texts)
                
                # Save index
                self._save_index()
                
                build_time = time.time() - start_time
                logger.info(f"Index built successfully in {build_time:.2f}s: "
                          f"{len(all_texts)} chunks, {self.tfidf_matrix.shape[1]} features")
                
                # Clear cache since index has changed
                self.cache.clear()
                
                return True
                
        except Exception as e:
            logger.error(f"Error building search index: {e}")
            return False
    
    def _save_index(self):
        """Save search index to disk."""
        try:
            # Convert numpy types to Python native types for JSON serialization
            def convert_numpy_types(obj):
                if hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                else:
                    return obj
            
            index_data = {
                'vectorizer_vocab': convert_numpy_types(self.vectorizer.vocabulary_) if self.vectorizer else {},
                'vectorizer_params': {
                    'max_features': int(self.vectorizer.max_features) if self.vectorizer else self.max_features,
                    'ngram_range': list(self.vectorizer.ngram_range) if self.vectorizer else (1, 2),
                    'stop_words': list(self.vectorizer.stop_words) if self.vectorizer and self.vectorizer.stop_words else None,
                    'lowercase': bool(self.vectorizer.lowercase) if self.vectorizer else True,
                    'token_pattern': str(self.vectorizer.token_pattern) if self.vectorizer else r'\b[a-zA-Z]{2,}\b',
                    'max_df': float(self.vectorizer.max_df) if self.vectorizer else 0.8,
                    'min_df': int(self.vectorizer.min_df) if self.vectorizer else 1
                },
                'chunk_metadata': convert_numpy_types(self.chunk_metadata),
                'doc_chunks_map': convert_numpy_types(self.doc_chunks_map),
                'index_version': '1.0',
                'created_at': datetime.now().isoformat()
            }
            
            # Save metadata
            with open(self.index_path / 'search_index.json', 'w') as f:
                json.dump(index_data, f, indent=2)
            
            # Save TF-IDF matrix
            if self.tfidf_matrix is not None:
                sparse.save_npz(self.index_path / 'tfidf_matrix.npz', self.tfidf_matrix)
            
            logger.info("Search index saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving search index: {e}")
            raise
    
    def _load_index(self) -> bool:
        """Load search index from disk."""
        try:
            index_file = self.index_path / 'search_index.json'
            matrix_file = self.index_path / 'tfidf_matrix.npz'
            
            if not (index_file.exists() and matrix_file.exists()):
                logger.info("No existing search index found")
                return False
            
            # Load metadata
            with open(index_file, 'r') as f:
                index_data = json.load(f)
            
            # Restore vectorizer
            vocab = index_data['vectorizer_vocab']
            params = index_data['vectorizer_params']
            
            self.vectorizer = TfidfVectorizer(
                vocabulary=vocab,
                max_features=params['max_features'],
                ngram_range=tuple(params['ngram_range']),
                stop_words=params['stop_words'],
                lowercase=params['lowercase'],
                token_pattern=params['token_pattern'],
                max_df=params['max_df'],
                min_df=params['min_df']
            )
            
            # Load matrix
            self.tfidf_matrix = sparse.load_npz(matrix_file)
            
            # Load metadata
            self.chunk_metadata = index_data['chunk_metadata']
            self.doc_chunks_map = index_data['doc_chunks_map']
            
            logger.info(f"Loaded search index: {len(self.chunk_metadata)} chunks, "
                       f"{self.tfidf_matrix.shape[1]} features")
            return True
            
        except Exception as e:
            logger.error(f"Error loading search index: {e}")
            return False
    
    def update_index_with_document(self, doc_id: str) -> bool:
        """
        Incrementally update index with a new document.
        
        Args:
            doc_id: ID of the document to add to index
            
        Returns:
            True if update was successful
        """
        try:
            with self.lock:
                # For now, do a full rebuild for simplicity
                # In production, this could be optimized for incremental updates
                logger.info(f"Updating index with document {doc_id}")
                return self.build_index(force_rebuild=True)
                
        except Exception as e:
            logger.error(f"Error updating index with document {doc_id}: {e}")
            return False
    
    def preprocess_query(self, query: str) -> str:
        """Preprocess search query."""
        return self._preprocess_text(query)
    
    def _calculate_exact_match_boost(self, query: str, text: str) -> float:
        """Calculate boost for exact phrase matches."""
        query_lower = query.lower()
        text_lower = text.lower()
        
        if query_lower in text_lower:
            # Boost based on query length and position
            query_words = len(query.split())
            if query_words > 1:
                return 0.3  # 30% boost for multi-word exact matches
            else:
                return 0.1  # 10% boost for single word exact matches
        return 0.0
    
    def _calculate_recency_score(self, upload_date: str, max_age_days: int = 365) -> float:
        """Calculate recency score based on document age."""
        try:
            upload_dt = datetime.fromisoformat(upload_date.replace('Z', '+00:00'))
            age_days = (datetime.now().replace(tzinfo=upload_dt.tzinfo) - upload_dt).days
            
            if age_days < 0:
                age_days = 0
            
            # Decay factor: newer documents get higher scores
            decay_factor = max(0.5, 1.0 - (age_days / max_age_days))
            return decay_factor
            
        except Exception as e:
            logger.warning(f"Error calculating recency score: {e}")
            return 0.8  # Default neutral score
    
    def _calculate_position_score(self, page_number: int, total_pages: int) -> float:
        """Calculate position score - earlier pages often more important."""
        if total_pages <= 1:
            return 1.0
        
        # Earlier pages get higher scores
        position_ratio = (total_pages - page_number + 1) / total_pages
        return 0.7 + 0.3 * position_ratio  # Score between 0.7 and 1.0
    
    def _calculate_section_match_boost(self, query: str, section_title: str) -> float:
        """Calculate boost for query terms appearing in section titles."""
        if not section_title or section_title == "Content":
            return 0.0
        
        query_terms = set(query.lower().split())
        section_terms = set(section_title.lower().split())
        
        # Calculate overlap
        overlap = len(query_terms.intersection(section_terms))
        if overlap > 0:
            return min(0.2 * overlap, 0.5)  # Up to 50% boost
        
        return 0.0
    
    def _calculate_hybrid_score(self, tfidf_score: float, query: str, metadata: Dict[str, Any]) -> float:
        """Calculate hybrid relevance score combining multiple signals."""
        try:
            # Base TF-IDF score
            base_score = tfidf_score
            
            # Calculate boosts
            exact_match_boost = self._calculate_exact_match_boost(query, metadata['text'])
            recency_score = self._calculate_recency_score(metadata['upload_date'])
            
            # Get total pages for position scoring
            doc_result = self.storage.load_document(metadata['doc_id'])
            total_pages = doc_result.metadata.total_pages if doc_result else 1
            position_score = self._calculate_position_score(metadata['page_number'], total_pages)
            
            section_match_boost = self._calculate_section_match_boost(query, metadata['section_title'])
            
            # Combine scores
            boosted_score = base_score * (1 + exact_match_boost + section_match_boost)
            final_score = boosted_score * recency_score * position_score
            
            return final_score
            
        except Exception as e:
            logger.warning(f"Error calculating hybrid score: {e}")
            return tfidf_score  # Fallback to TF-IDF score
    
    def _generate_snippet(self, text: str, query: str, max_length: int = 200) -> str:
        """Generate highlighted snippet around query matches."""
        query_terms = query.lower().split()
        text_lower = text.lower()
        
        # Find the best match position
        best_pos = 0
        best_score = 0
        
        for term in query_terms:
            pos = text_lower.find(term)
            if pos != -1:
                # Score based on term length and position
                score = len(term) / (pos + 1)
                if score > best_score:
                    best_score = score
                    best_pos = pos
        
        # Extract snippet around best position
        start = max(0, best_pos - max_length // 2)
        end = min(len(text), start + max_length)
        
        # Adjust to word boundaries
        if start > 0:
            while start < len(text) and not text[start].isspace():
                start += 1
        
        if end < len(text):
            while end > start and not text[end].isspace():
                end -= 1
        
        snippet = text[start:end].strip()
        
        # Add ellipsis if truncated
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        
        return snippet
    
    def _get_context(self, chunk_idx: int, context_size: int = 100) -> Tuple[str, str]:
        """Get surrounding context for a chunk."""
        try:
            current_meta = self.chunk_metadata[chunk_idx]
            doc_id = current_meta['doc_id']
            
            # Get chunks from same document
            doc_chunk_indices = self.doc_chunks_map.get(doc_id, [])
            try:
                current_pos = doc_chunk_indices.index(chunk_idx)
            except ValueError:
                return "", ""
            
            context_before = ""
            context_after = ""
            
            # Get context before
            if current_pos > 0:
                prev_meta = self.chunk_metadata[doc_chunk_indices[current_pos - 1]]
                context_before = prev_meta['text'][-context_size:] if len(prev_meta['text']) > context_size else prev_meta['text']
            
            # Get context after
            if current_pos < len(doc_chunk_indices) - 1:
                next_meta = self.chunk_metadata[doc_chunk_indices[current_pos + 1]]
                context_after = next_meta['text'][:context_size] if len(next_meta['text']) > context_size else next_meta['text']
            
            return context_before, context_after
            
        except Exception as e:
            logger.warning(f"Error getting context for chunk {chunk_idx}: {e}")
            return "", ""
    
    def search_similar_chunks(self, query: str, top_k: int = 5, 
                            filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Find most relevant chunks using TF-IDF similarity and hybrid scoring.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional filters (doc_id, filename, page_range, etc.)
            
        Returns:
            List of SearchResult objects ranked by relevance
        """
        if filters is None:
            filters = {}
        
        try:
            start_time = time.time()
            
            # Check cache
            query_hash = self._generate_query_hash(query, top_k, filters)
            cached_results = self.cache.get(query_hash)
            if cached_results:
                self.search_stats['cache_hits'] += 1
                return cached_results[:top_k]
            
            with self.lock:
                if self.tfidf_matrix is None or len(self.chunk_metadata) == 0:
                    logger.warning("Search index not built. Building now...")
                    if not self.build_index():
                        return []
                
                # Preprocess query
                processed_query = self.preprocess_query(query)
                if not processed_query:
                    return []
                
                # Transform query to TF-IDF vector
                query_vector = self.vectorizer.transform([processed_query])
                
                # Calculate cosine similarities
                similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                
                # Get top candidates (more than needed for filtering)
                candidate_count = min(top_k * 3, len(similarities))
                top_indices = np.argsort(similarities)[-candidate_count:][::-1]
                
                # Apply filters and calculate hybrid scores
                results = []
                seen_docs = set()
                
                for idx in top_indices:
                    if similarities[idx] < 0.01:  # Minimum similarity threshold
                        continue
                    
                    metadata = self.chunk_metadata[idx]
                    
                    # Apply filters
                    if filters:
                        if 'doc_id' in filters and metadata['doc_id'] != filters['doc_id']:
                            continue
                        if 'filename' in filters and filters['filename'] not in metadata['filename']:
                            continue
                        if 'min_page' in filters and metadata['page_number'] < filters['min_page']:
                            continue
                        if 'max_page' in filters and metadata['page_number'] > filters['max_page']:
                            continue
                    
                    # Avoid too many results from same document
                    doc_id = metadata['doc_id']
                    if doc_id in seen_docs:
                        doc_count = sum(1 for r in results if r.doc_id == doc_id)
                        if doc_count >= 2:  # Max 2 chunks per document in top results
                            continue
                    seen_docs.add(doc_id)
                    
                    # Calculate hybrid score
                    hybrid_score = self._calculate_hybrid_score(similarities[idx], query, metadata)
                    
                    # Generate snippet and context
                    snippet = self._generate_snippet(metadata['text'], query)
                    context_before, context_after = self._get_context(idx)
                    
                    # Create search result
                    result = SearchResult(
                        chunk_id=metadata['chunk_id'],
                        doc_id=metadata['doc_id'],
                        text=metadata['text'],
                        similarity_score=float(similarities[idx]),
                        page_number=metadata['page_number'],
                        section_title=metadata['section_title'],
                        filename=metadata['filename'],
                        snippet=snippet,
                        context_before=context_before,
                        context_after=context_after,
                        combined_score=hybrid_score
                    )
                    
                    results.append(result)
                
                # Sort by combined score and limit results
                results.sort(key=lambda x: x.combined_score, reverse=True)
                final_results = results[:top_k]
                
                # Cache results
                self.cache.put(query_hash, final_results)
                
                # Update stats
                response_time = (time.time() - start_time) * 1000  # ms
                self.search_stats['total_searches'] += 1
                self.search_stats['total_response_time'] += response_time
                
                logger.info(f"Search completed in {response_time:.1f}ms: "
                          f"'{query}' -> {len(final_results)} results")
                
                return final_results
                
        except Exception as e:
            logger.error(f"Error in search: {e}")
            return []
    
    def rank_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Re-rank search results (already done in search_similar_chunks)."""
        return sorted(results, key=lambda x: x.combined_score, reverse=True)
    
    def filter_results(self, results: List[SearchResult], 
                      min_score: float = 0.1,
                      deduplicate: bool = True) -> List[SearchResult]:
        """Filter and post-process search results."""
        filtered = []
        seen_texts = set()
        
        for result in results:
            # Apply minimum score threshold
            if result.combined_score < min_score:
                continue
            
            # Deduplicate similar texts
            if deduplicate:
                text_hash = hashlib.sha256(result.text.encode()).hexdigest()[:16]
                if text_hash in seen_texts:
                    continue
                seen_texts.add(text_hash)
            
            filtered.append(result)
        
        return filtered
    
    def get_search_stats(self) -> SearchStats:
        """Get search engine statistics."""
        try:
            avg_response_time = 0.0
            if self.search_stats['total_searches'] > 0:
                avg_response_time = (self.search_stats['total_response_time'] / 
                                   self.search_stats['total_searches'])
            
            cache_hit_rate = 0.0
            if self.search_stats['total_searches'] > 0:
                cache_hit_rate = (self.search_stats['cache_hits'] / 
                                self.search_stats['total_searches'])
            
            index_size = 0
            if self.index_path.exists():
                for file_path in self.index_path.glob('*'):
                    index_size += file_path.stat().st_size
            
            last_update = "Never"
            index_file = self.index_path / 'search_index.json'
            if index_file.exists():
                last_update = datetime.fromtimestamp(index_file.stat().st_mtime).isoformat()
            
            return SearchStats(
                total_searches=self.search_stats['total_searches'],
                average_response_time_ms=avg_response_time,
                cache_hit_rate=cache_hit_rate,
                index_size_bytes=index_size,
                total_documents_indexed=len(self.doc_chunks_map),
                last_index_update=last_update
            )
            
        except Exception as e:
            logger.error(f"Error getting search stats: {e}")
            return SearchStats(0, 0.0, 0.0, 0, 0, "Error")
    
    def clear_cache(self):
        """Clear search result cache."""
        self.cache.clear()
        logger.info("Search cache cleared")
    
    def rebuild_index(self) -> bool:
        """Force a complete rebuild of the search index."""
        logger.info("Rebuilding search index...")
        self.clear_cache()
        return self.build_index(force_rebuild=True) 