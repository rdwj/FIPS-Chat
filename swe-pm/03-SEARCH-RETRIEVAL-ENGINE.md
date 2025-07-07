# Chat Guide 03: Search & Retrieval Engine

## Objective
Implement a file-based similarity search engine using TF-IDF and other FIPS-compliant techniques to find relevant document chunks for RAG queries.

## Prerequisites
- Chat Guide 01 completed (document processing foundation)
- Chat Guide 02 completed (file-based storage system)
- Documents can be stored and retrieved efficiently
- Feature branch: `feature/rag-implementation`

## Goals for This Chat Session
1. Implement TF-IDF similarity search for document chunks
2. Create hybrid search combining multiple relevance signals
3. Build efficient search indexing and query processing
4. Add relevance scoring and ranking algorithms
5. Implement search result filtering and post-processing

## Tasks to Complete

### 1. TF-IDF Search Engine
**What to Ask For:**
```
Create rag/search_engine.py with TFIDFSearchEngine class:
- Build TF-IDF vectors from document chunks
- Implement cosine similarity search
- Handle incremental index updates when new documents added
- Memory-efficient search for large document sets
- FIPS-compliant implementation (no MD5 hashing)
```

### 2. Search Index Management
**What to Ask For:**
```
Implement search index creation and management:
- build_search_index() method to create TF-IDF matrix
- update_index() for incremental updates
- save_index() and load_index() for persistence
- rebuild_index() for full reindexing
- Memory-efficient matrix operations
```

### 3. Query Processing
**What to Ask For:**
```
Add query processing and search capabilities:
- preprocess_query() for text cleaning and normalization
- search_similar_chunks() for finding relevant content
- rank_results() for relevance scoring
- filter_results() for post-processing
- Handle various query types (keywords, phrases, questions)
```

### 4. Hybrid Search Implementation
**What to Ask For:**
```
Enhance search with multiple relevance signals:
- TF-IDF cosine similarity as primary signal
- Exact phrase matching boost
- Document recency scoring
- Page position relevance (earlier pages often more important)
- Section title matching (headings, abstracts)
- Combine signals with weighted scoring
```

### 5. Search Result Processing
**What to Ask For:**
```
Implement search result processing and formatting:
- Deduplicate similar chunks from same document
- Context window expansion around relevant chunks
- Snippet generation with highlighting
- Source attribution with page numbers and sections
- Relevance threshold filtering
```

### 6. Performance Optimization
**What to Ask For:**
```
Optimize search performance for memory-constrained environment:
- Lazy loading of search indexes
- Chunk-level caching for frequent searches
- Query result caching with TTL
- Memory usage monitoring during search
- Batch processing for multiple queries
```

### 7. Testing
**What to Ask For:**
```
Create tests/test_search_engine.py:
- Test TF-IDF index building and search
- Test hybrid search relevance scoring
- Test query processing and result formatting
- Test performance with target document set (300 pages)
- Test memory usage stays within limits
- Verify FIPS compliance throughout
```

## Expected Outputs After This Chat
- [ ] `rag/search_engine.py` with TFIDFSearchEngine class
- [ ] TF-IDF similarity search working efficiently
- [ ] Hybrid search with multiple relevance signals
- [ ] Query processing and result formatting
- [ ] Search index persistence and management
- [ ] Performance optimizations for memory constraints
- [ ] Comprehensive search engine tests

## Key Implementation Details

### TFIDFSearchEngine Class Interface
```python
class TFIDFSearchEngine:
    def __init__(self, storage: FileStorage, max_features: int = 5000):
        self.storage = storage
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = None
        self.chunk_metadata = []
        
    def build_index(self) -> None:
        # Build TF-IDF matrix from all document chunks
        
    def search_similar_chunks(self, query: str, top_k: int = 5) -> List[SearchResult]:
        # Find most relevant chunks using cosine similarity
        
    def update_index_with_document(self, doc_id: str) -> None:
        # Incrementally update index with new document
```

### Search Result Structure
```python
@dataclass
class SearchResult:
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
```

### Hybrid Scoring Algorithm
```python
def calculate_hybrid_score(
    tfidf_score: float,
    exact_match_boost: float,
    recency_score: float,
    position_score: float,
    section_match_boost: float
) -> float:
    """Combine multiple relevance signals"""
    base_score = tfidf_score
    boosted_score = base_score * (1 + exact_match_boost + section_match_boost)
    final_score = boosted_score * recency_score * position_score
    return final_score
```

### Search Performance Optimizations
```python
class SearchOptimizer:
    def __init__(self, cache_size: int = 100):
        self.query_cache = LRUCache(cache_size)
        self.index_cache = {}
        
    def cached_search(self, query: str, top_k: int) -> List[SearchResult]:
        # Cache search results for repeated queries
        
    def lazy_load_index(self) -> None:
        # Load search index only when needed
        
    def memory_efficient_search(self, query: str) -> List[SearchResult]:
        # Process search with minimal memory footprint
```

### Index Persistence
```python
# Save search index to disk
{
    "vectorizer_vocab": {...},
    "tfidf_matrix": "compressed_sparse_matrix.npz",
    "chunk_metadata": [
        {
            "chunk_id": "chunk_001",
            "doc_id": "doc_abcd1234", 
            "text_length": 987,
            "page_number": 1,
            "section": "Introduction"
        }
    ],
    "index_version": "1.0",
    "created_at": "2024-01-01T00:00:00Z"
}
```

## Search Quality Requirements
- **Relevance**: Return contextually relevant chunks for queries
- **Precision**: Avoid irrelevant results with appropriate thresholds
- **Coverage**: Find relevant content across all documents
- **Speed**: Search response under 2 seconds for typical queries
- **Memory**: Stay within 100MB memory limit during search

## Success Criteria for This Chat
- ✅ TF-IDF search finds relevant chunks for various query types
- ✅ Hybrid scoring improves relevance over basic TF-IDF
- ✅ Search index builds and persists correctly
- ✅ Memory usage stays within constraints during search
- ✅ Search performance meets response time requirements
- ✅ All search operations remain FIPS-compliant

## Query Types to Support
1. **Keyword queries**: "machine learning algorithms"
2. **Question queries**: "What are the benefits of AI?"
3. **Phrase queries**: "neural network architecture"
4. **Topic queries**: "data privacy regulations"
5. **Complex queries**: "challenges in implementing GDPR compliance"

## Next Chat Guide
After completing the search engine, proceed to **Chat Guide 04: RAG Pipeline Integration** to connect the search functionality with the chat interface.

## Notes for Implementation
- Use scikit-learn's TfidfVectorizer for reliability
- Implement proper text preprocessing (lowercase, punctuation, stopwords)
- Consider document structure in relevance scoring
- Cache search results to improve response times
- Monitor memory usage during index building and search
- Test with realistic queries from the target domain