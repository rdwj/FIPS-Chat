#!/usr/bin/env python3
"""
Example usage of the RAG system with UI components.
This demonstrates how to use the complete RAG pipeline with the new UI.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.document_processor import DoclingProcessor, RAGConfig
from rag.file_storage import FileStorage
from rag.search_engine import TFIDFSearchEngine
from rag.rag_pipeline import RAGPipeline


def main():
    """Demonstrate RAG system usage."""
    print("🔍 RAG System Example Usage")
    print("=" * 50)
    
    # Initialize components
    print("1. Initializing RAG components...")
    
    # Storage
    storage = FileStorage("example_rag_storage", max_memory_mb=100)
    
    # Document processor
    config = RAGConfig(
        chunk_size=1000,
        chunk_overlap=200,
        min_chunk_size=100,
        max_chunk_size=2000
    )
    processor = DoclingProcessor(config=config, storage=storage, auto_store=True)
    
    # Search engine
    search_engine = TFIDFSearchEngine(storage)
    
    print("✅ RAG components initialized")
    
    # Example text processing (since PDF might not be available)
    print("\n2. Processing example text...")
    
    example_text = """
    FIPS (Federal Information Processing Standards) are publicly announced standards developed by the National Institute of Standards and Technology (NIST) for use in computer systems by non-military government agencies and government contractors.

    FIPS 140-2 is a security standard that specifies the security requirements that will be satisfied by a cryptographic module. The standard provides four increasing levels of security: Level 1, Level 2, Level 3, and Level 4.

    FIPS compliance is mandatory for federal agencies and is often required for contractors working with federal agencies. The standards cover areas such as cryptographic algorithms, key management, and physical security requirements.

    Organizations implementing FIPS compliant systems must ensure that all cryptographic operations meet the specified requirements and that the systems are properly validated and certified.
    """
    
    try:
        # Process the example text
        doc_result = processor.process_text(example_text, "FIPS_Information.txt")
        print(f"✅ Processed document: {doc_result.metadata.doc_id[:12]}...")
        print(f"   - {doc_result.metadata.total_chunks} chunks created")
        
        # Build search index
        print("\n3. Building search index...")
        search_engine.build_index_from_storage()
        
        # Get index stats
        stats = search_engine.get_index_stats()
        print(f"✅ Search index built")
        print(f"   - {stats.get('total_documents', 0)} documents indexed")
        print(f"   - {stats.get('total_terms', 0)} unique terms")
        
        # Example search
        print("\n4. Testing search functionality...")
        search_results = search_engine.search_similar_chunks(
            query="What are FIPS security levels?",
            top_k=3
        )
        
        print(f"✅ Found {len(search_results)} relevant chunks")
        for i, result in enumerate(search_results):
            print(f"   Result {i+1}: Score {result.combined_score:.3f}")
            print(f"   Snippet: {result.snippet[:100]}...")
        
        print("\n5. Storage statistics...")
        storage_stats = storage.get_storage_stats()
        print(f"✅ Storage info:")
        print(f"   - Documents: {storage_stats.total_documents}")
        print(f"   - Chunks: {storage_stats.total_chunks}")
        print(f"   - Disk usage: {storage_stats.total_disk_usage_bytes / 1024:.1f} KB")
        print(f"   - Memory usage: {storage_stats.memory_usage_bytes / 1024:.1f} KB")
        print(f"   - Cache hit rate: {storage_stats.cache_hit_rate:.2%}")
        
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        return 1
    
    print("\n🎉 RAG system example completed successfully!")
    print("\nTo use the UI components:")
    print("1. Run: streamlit run app.py")
    print("2. Configure your API endpoint in the sidebar")
    print("3. Go to the 'Documents' tab to upload PDFs")
    print("4. Use the 'RAG Chat' tab for enhanced conversations")
    
    return 0


if __name__ == "__main__":
    exit(main())