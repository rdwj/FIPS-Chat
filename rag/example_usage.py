"""
Example usage of the complete RAG Pipeline system.
Demonstrates how to initialize and use RAG for document-enhanced AI chat.
"""

import logging
from pathlib import Path
from typing import Optional

from .file_storage import FileStorage
from .search_engine import TFIDFSearchEngine
from .rag_pipeline import RAGPipeline, RAGQualityController
from ai_client import UnifiedAPIClient, APIEndpoint, APIType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_rag_system(
    storage_path: str = "rag_storage",
    api_endpoint_url: str = "http://localhost:11434/v1",
    api_key: Optional[str] = None,
    api_type: str = "openai_compatible"
) -> RAGPipeline:
    """
    Set up complete RAG system with all components.
    
    Args:
        storage_path: Path for document storage
        api_endpoint_url: AI API endpoint URL
        api_key: API key for authentication
        api_type: Type of API ("openai_compatible", "anthropic", etc.)
    
    Returns:
        Configured RAGPipeline instance
    """
    logger.info("Setting up RAG system...")
    
    # Initialize file storage
    storage = FileStorage(storage_path, max_memory_mb=100)
    logger.info(f"File storage initialized at {storage_path}")
    
    # Initialize search engine
    search_engine = TFIDFSearchEngine(
        storage=storage,
        max_features=5000,
        cache_size=100
    )
    logger.info("Search engine initialized")
    
    # Build search index if documents exist
    documents = storage.list_documents()
    if documents:
        logger.info(f"Building search index for {len(documents)} documents...")
        success = search_engine.build_index()
        if success:
            logger.info("Search index built successfully")
        else:
            logger.warning("Failed to build search index")
    else:
        logger.info("No documents found - search index will be built when documents are added")
    
    # Initialize AI client
    api_type_enum = APIType.OPENAI_COMPATIBLE
    if api_type == "anthropic":
        api_type_enum = APIType.ANTHROPIC
    elif api_type == "agentic_pipeline":
        api_type_enum = APIType.AGENTIC_PIPELINE
    
    endpoint = APIEndpoint(
        url=api_endpoint_url,
        api_key=api_key,
        api_type=api_type_enum
    )
    
    ai_client = UnifiedAPIClient(endpoint)
    logger.info(f"AI client initialized for {api_type}")
    
    # Test AI client connection
    success, message = ai_client.test_connection()
    if success:
        logger.info(f"AI client connection successful: {message}")
    else:
        logger.warning(f"AI client connection failed: {message}")
    
    # Initialize quality controller with custom settings
    quality_controller = RAGQualityController(
        min_relevance=0.1,
        min_context_length=50
    )
    
    # Create RAG pipeline
    rag_pipeline = RAGPipeline(
        storage=storage,
        search_engine=search_engine,
        ai_client=ai_client,
        max_context_tokens=3000,
        quality_controller=quality_controller
    )
    
    logger.info("RAG pipeline initialized successfully")
    return rag_pipeline


def demo_rag_query(rag_pipeline: RAGPipeline, query: str, model: str = "llama3.2:latest"):
    """
    Demonstrate RAG query processing.
    
    Args:
        rag_pipeline: Configured RAG pipeline
        query: User query
        model: AI model to use
    """
    logger.info(f"Processing RAG query: {query}")
    
    try:
        # Process query through RAG pipeline
        response = rag_pipeline.process_rag_query(
            query=query,
            model=model,
            max_sources=5,
            relevance_threshold=0.1,
            stream=True,
            temperature=0.7,
            max_tokens=1000
        )
        
        # Display results
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        print(f"RAG Mode: {response.rag_mode}")
        print(f"Processing Time: {response.processing_time:.2f} seconds")
        print(f"Number of Sources: {len(response.sources)}")
        print(f"\nResponse:")
        print(response.response)
        
        if response.sources:
            print(f"\nSources Used:")
            for i, source in enumerate(response.sources, 1):
                print(f"  {i}. {source.document} (Page {source.page_number})")
                print(f"     Section: {source.section_title}")
                print(f"     Relevance: {source.relevance_score:.3f}")
                print(f"     Excerpt: {source.excerpt[:100]}...")
        
        # Show pipeline statistics
        stats = rag_pipeline.get_pipeline_stats()
        print(f"\nPipeline Statistics:")
        print(f"  Total Queries: {stats['total_queries']}")
        print(f"  Full RAG Rate: {stats['full_rag_rate']:.1%}")
        print(f"  Partial RAG Rate: {stats['partial_rag_rate']:.1%}")
        print(f"  Fallback Rate: {stats['fallback_rate']:.1%}")
        print(f"  Average Processing Time: {stats['avg_processing_time']:.2f}s")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing RAG query: {e}")
        raise


def main():
    """Main example function."""
    print("RAG Pipeline Integration Example")
    print("=" * 40)
    
    # Setup RAG system
    rag_pipeline = setup_rag_system(
        storage_path="rag_storage",
        api_endpoint_url="http://localhost:11434/v1",  # Ollama default
        api_key=None,  # No API key needed for local Ollama
        api_type="openai_compatible"
    )
    
    # Example queries to test different scenarios
    example_queries = [
        "What is machine learning?",
        "How does artificial intelligence work?",
        "What are the main types of neural networks?",
        "Explain the difference between supervised and unsupervised learning",
        "What are the applications of natural language processing?"
    ]
    
    print("\nRunning example queries...")
    
    for query in example_queries:
        try:
            response = demo_rag_query(rag_pipeline, query)
            input("\nPress Enter to continue to next query...")
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
            break
        except Exception as e:
            print(f"Error with query '{query}': {e}")
            continue
    
    print("\nRAG Pipeline demo completed!")


if __name__ == "__main__":
    main() 