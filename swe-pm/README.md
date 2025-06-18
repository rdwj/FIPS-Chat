# FIPS Chat RAG Implementation - Chat Guide Index

This directory contains a complete implementation plan for adding File-Based RAG (Retrieval-Augmented Generation) functionality to FIPS Chat. The implementation is broken down into 8 focused chat sessions that build incrementally.

## Quick Start

1. **Create feature branch**: `git checkout -b feature/rag-implementation`
2. **Follow chat guides sequentially**: Start with Chat Guide 01
3. **Test after each guide**: Validate functionality before proceeding
4. **Deploy**: Use Chat Guide 08 for OpenShift deployment

## Chat Guide Overview

| Guide | Focus | Estimated Time | Key Outputs |
|-------|-------|----------------|-------------|
| [01 - Document Processing](01-DOCUMENT-PROCESSING-FOUNDATION.md) | Docling integration, text extraction, chunking | 2-3 hours | Document processor, FIPS validation |
| [02 - File Storage](02-FILE-BASED-STORAGE-SYSTEM.md) | File-based storage, memory management | 2-3 hours | Storage system, document indexing |
| [03 - Search Engine](03-SEARCH-RETRIEVAL-ENGINE.md) | TF-IDF similarity search, relevance scoring | 2-3 hours | Search engine, hybrid scoring |
| [04 - RAG Pipeline](04-RAG-PIPELINE-INTEGRATION.md) | End-to-end RAG, context management | 2-3 hours | Complete RAG pipeline, citations |
| [05 - UI Components](05-USER-INTERFACE-COMPONENTS.md) | Document upload, RAG chat interface | 2-3 hours | User interface, document management |
| [06 - Configuration](06-CONFIGURATION-SETTINGS.md) | Environment variables, settings management | 1-2 hours | Configuration system, deployment config |
| [07 - Testing](07-TESTING-VALIDATION.md) | Unit tests, integration tests, FIPS validation | 2-3 hours | Comprehensive test suite |
| [08 - Deployment](08-OPENSHIFT-DEPLOYMENT.md) | OpenShift manifests, production deployment | 1-2 hours | Production deployment guide |

**Total Estimated Time**: 14-20 hours

## Architecture Overview

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Document Upload   │    │   File Storage      │    │   Search Engine     │
│   (UI Component)    │───▶│   (JSON + Indexes)  │───▶│   (TF-IDF + Hybrid) │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
           │                           │                           │
           ▼                           ▼                           ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│ Docling Processor   │    │   Memory Manager    │    │   RAG Pipeline      │
│ (Text Extraction)   │    │   (LRU Cache)       │    │   (Context + Gen)   │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
           │                           │                           │
           └───────────────────────────┼───────────────────────────┘
                                       ▼
                           ┌─────────────────────┐
                           │   Chat Interface    │
                           │   (RAG Enhanced)    │
                           └─────────────────────┘
```

## Key Design Decisions

### Why File-Based RAG?
- **FIPS Compliance**: Avoids MD5 issues in vector databases like LanceDB/Milvus
- **Memory Efficient**: Suitable for demo environment constraints (100MB limit)
- **Simple Deployment**: No complex database dependencies
- **Demo Scale**: Perfect for 300 pages across 75 PDFs

### Technology Choices
- **Docling**: Robust PDF text extraction with structure preservation
- **TF-IDF + scikit-learn**: Proven similarity search, FIPS-compliant
- **JSON Storage**: Human-readable, debuggable, simple persistence
- **SHA-256 Hashing**: FIPS-compliant document fingerprinting

## Prerequisites

### System Requirements
- Python 3.9+ with FIPS-enabled OpenSSL
- OpenShift cluster with persistent storage
- 10Gi storage for RAG data, 2Gi memory for processing
- FIPS 140-2 enabled environment for production

### Dependencies to Add
```bash
# Core RAG dependencies
docling>=1.0.0
docling-core>=1.0.0  
docling-parse>=1.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
nltk>=3.8.1
```

## Target Use Case

**Demo Environment Specifications:**
- 75 PDF documents maximum
- 300 total pages across all documents
- 100MB memory limit for RAG operations
- 5-second response time for RAG queries
- Full FIPS 140-2 compliance required

## Implementation Strategy

### Phase 1: Foundation (Guides 01-02)
Build document processing and storage without AI integration. Focus on FIPS compliance and memory efficiency.

### Phase 2: Search (Guide 03)
Add similarity search capabilities. Test with realistic document sets.

### Phase 3: Integration (Guides 04-05)
Connect search to AI generation. Build user interface for document management.

### Phase 4: Production (Guides 06-08)
Add configuration, testing, and deployment capabilities for production use.

## Success Metrics

- ✅ **Functionality**: Upload documents, generate embeddings, get RAG responses
- ✅ **Performance**: < 5 second response time, < 100MB memory usage
- ✅ **Scale**: Handle 75 PDFs (300 pages) simultaneously
- ✅ **Compliance**: Pass FIPS validation in enabled environment
- ✅ **Deployment**: Successfully deploy to OpenShift with persistence
- ✅ **Usability**: Intuitive document upload and RAG chat interface

## Common Issues and Solutions

### FIPS Compliance Issues
- **Problem**: MD5 usage causing kernel crashes
- **Solution**: Use SHA-256 for all hashing, validate all dependencies
- **Testing**: Run with `OPENSSL_FIPS=1` throughout development

### Memory Constraints
- **Problem**: Memory usage exceeding limits
- **Solution**: Implement LRU caching, lazy loading, stream processing
- **Monitoring**: Track memory usage during document processing

### Search Quality
- **Problem**: Poor relevance in search results
- **Solution**: Implement hybrid scoring, tune relevance thresholds
- **Validation**: Test with realistic queries and documents

## Getting Help

### During Implementation
- Follow each chat guide step-by-step
- Test functionality after each guide
- Validate FIPS compliance early and often
- Monitor memory usage throughout

### Troubleshooting
- Check FIPS mode: `echo $OPENSSL_FIPS`
- Validate storage: `ls -la /mnt/rag-storage`
- Test document processing: Upload single PDF first
- Monitor memory: `docker stats` or `oc top pods`

### After Completion
- Merge feature branch after successful deployment
- Update documentation with any deployment-specific issues
- Create user training materials for RAG functionality
- Plan regular maintenance and backup procedures

## File Structure After Implementation

```
FIPS-Chat/
├── swe-pm/                          # This directory (implementation guides)
├── rag/                             # New RAG module
│   ├── __init__.py
│   ├── document_processor.py        # Docling integration
│   ├── file_storage.py             # File-based storage system
│   ├── search_engine.py            # TF-IDF similarity search
│   └── rag_pipeline.py             # End-to-end RAG processing
├── ui_components/
│   ├── document_interface.py        # Document upload/management UI
│   └── rag_chat_interface.py       # RAG-enhanced chat interface
├── tests/
│   ├── test_rag_*.py               # Comprehensive RAG tests
│   └── fixtures/                   # Test data and sample PDFs
├── openshift/
│   ├── rag-pvc.yaml                # Persistent volume for RAG storage
│   └── rag-configmap.yaml          # RAG configuration
└── rag_storage/                     # Runtime RAG data (gitignored)
    ├── documents/                   # Document metadata and chunks
    ├── indexes/                     # Search indexes
    └── config/                      # RAG runtime configuration
```

---

**Ready to start?** Begin with [Chat Guide 01: Document Processing Foundation](01-DOCUMENT-PROCESSING-FOUNDATION.md)

For questions or issues during implementation, refer to the specific chat guide or the troubleshooting sections in each guide.