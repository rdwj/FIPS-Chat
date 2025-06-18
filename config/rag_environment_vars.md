# RAG Configuration Environment Variables

This document provides a comprehensive reference for all RAG-related environment variables and configuration options.

## Core RAG Settings

### ENABLE_RAG
**Default:** `true`  
**Description:** Enable or disable RAG functionality  
**Valid Values:** `true`, `false`  
**Example:** `ENABLE_RAG=true`

### RAG_STORAGE_PATH
**Default:** `./rag_storage`  
**Description:** Path for RAG data storage  
**Valid Values:** Any valid filesystem path  
**Examples:**
- Development: `RAG_STORAGE_PATH=./dev_rag_storage`
- Demo: `RAG_STORAGE_PATH=/app/rag_storage`
- OpenShift: `RAG_STORAGE_PATH=/mnt/rag-storage`

## Memory and Performance

### RAG_MAX_MEMORY_MB
**Default:** `100`  
**Description:** Maximum memory usage for RAG operations (in MB)  
**Valid Range:** `50` - `1000`  
**Examples:**
- Development: `RAG_MAX_MEMORY_MB=200`
- Production: `RAG_MAX_MEMORY_MB=100`

### RAG_CACHE_SIZE
**Default:** `50`  
**Description:** Number of documents to cache in memory  
**Valid Range:** `10` - `1000`  
**Example:** `RAG_CACHE_SIZE=100`

## Document Processing

### RAG_CHUNK_SIZE
**Default:** `1000`  
**Description:** Text chunk size in characters  
**Valid Range:** `100` - `10000`  
**Examples:**
- Small chunks: `RAG_CHUNK_SIZE=800`
- Large chunks: `RAG_CHUNK_SIZE=1500`

### RAG_CHUNK_OVERLAP
**Default:** `200`  
**Description:** Overlap between text chunks in characters  
**Valid Range:** `0` - `chunk_size - 1`  
**Example:** `RAG_CHUNK_OVERLAP=150`

### RAG_MAX_DOCUMENT_SIZE_MB
**Default:** `10`  
**Description:** Maximum size for individual documents (in MB)  
**Valid Range:** `1` - `100`  
**Examples:**
- Small docs: `RAG_MAX_DOCUMENT_SIZE_MB=5`
- Large docs: `RAG_MAX_DOCUMENT_SIZE_MB=20`

### RAG_MAX_DOCUMENTS
**Default:** `100`  
**Description:** Maximum number of documents to store  
**Valid Range:** `1` - `10000`  
**Examples:**
- Development: `RAG_MAX_DOCUMENTS=20`
- Production: `RAG_MAX_DOCUMENTS=200`

## Search Configuration

### RAG_MAX_SEARCH_RESULTS
**Default:** `5`  
**Description:** Maximum number of search results per query  
**Valid Range:** `1` - `20`  
**Example:** `RAG_MAX_SEARCH_RESULTS=10`

### RAG_RELEVANCE_THRESHOLD
**Default:** `0.1`  
**Description:** Minimum relevance score for search results  
**Valid Range:** `0.0` - `1.0`  
**Examples:**
- Relaxed: `RAG_RELEVANCE_THRESHOLD=0.05`
- Strict: `RAG_RELEVANCE_THRESHOLD=0.3`

### RAG_TFIDF_MAX_FEATURES
**Default:** `5000`  
**Description:** Maximum number of TF-IDF features for search indexing  
**Valid Range:** `100` - `50000`  
**Examples:**
- Small index: `RAG_TFIDF_MAX_FEATURES=1000`
- Large index: `RAG_TFIDF_MAX_FEATURES=10000`

## FIPS Compliance

### OPENSSL_FIPS
**Default:** `0`  
**Description:** Enable FIPS 140-2 compliant operations  
**Valid Values:** `0`, `1`  
**Example:** `OPENSSL_FIPS=1`

### RAG_HASH_ALGORITHM
**Default:** `sha256`  
**Description:** Hash algorithm for FIPS-compliant operations  
**Valid Values:** `sha256`, `sha384`, `sha512` (when FIPS enabled)  
**Example:** `RAG_HASH_ALGORITHM=sha384`

## Demo Environment

### RAG_DEMO_MODE
**Default:** `false`  
**Description:** Enable demo-specific limits and features  
**Valid Values:** `true`, `false`  
**Example:** `RAG_DEMO_MODE=true`

### RAG_DEMO_MAX_PAGES
**Default:** `300`  
**Description:** Maximum pages to process in demo mode  
**Valid Range:** `100` - `10000`  
**Example:** `RAG_DEMO_MAX_PAGES=500`

### RAG_DEMO_MAX_PDFS
**Default:** `75`  
**Description:** Maximum PDFs to upload in demo mode  
**Valid Range:** `10` - `1000`  
**Example:** `RAG_DEMO_MAX_PDFS=100`

## Configuration Examples

### Development Environment
```bash
# Relaxed settings for development
ENABLE_RAG=true
RAG_STORAGE_PATH=./dev_rag_storage
RAG_MAX_MEMORY_MB=200
RAG_CHUNK_SIZE=800
RAG_MAX_DOCUMENTS=20
RAG_DEMO_MODE=true
OPENSSL_FIPS=0  # Relaxed for development
RAG_CACHE_SIZE=25
```

### Demo Environment
```bash
# Demo environment with limits
ENABLE_RAG=true
RAG_STORAGE_PATH=/app/rag_storage
RAG_MAX_MEMORY_MB=100
RAG_CHUNK_SIZE=1000
RAG_MAX_DOCUMENTS=100
RAG_DEMO_MODE=true
RAG_DEMO_MAX_PAGES=300
RAG_DEMO_MAX_PDFS=75
OPENSSL_FIPS=1
RAG_CACHE_SIZE=50
```

### OpenShift Production
```bash
# Production environment with FIPS compliance
ENABLE_RAG=true
RAG_STORAGE_PATH=/mnt/rag-storage
RAG_MAX_MEMORY_MB=100
RAG_CHUNK_SIZE=1000
RAG_MAX_DOCUMENTS=200
OPENSSL_FIPS=1
RAG_HASH_ALGORITHM=sha256
RAG_CACHE_SIZE=100
RAG_DEMO_MODE=false
```

## Configuration Validation

The system validates all configuration values and will provide error messages for invalid settings:

- **Memory limits:** Must be between 50MB and 1000MB
- **Chunk settings:** Chunk overlap must be less than chunk size
- **Document limits:** Document size must be between 1MB and 100MB
- **Search settings:** Relevance threshold must be between 0.0 and 1.0
- **FIPS compliance:** When enabled, only SHA-256, SHA-384, or SHA-512 algorithms are allowed
- **Demo mode:** Page and PDF limits must meet minimum requirements

## Configuration Precedence

Configuration values are loaded in the following order (later values override earlier ones):

1. **Default values** - Built-in defaults in the code
2. **Configuration file** - `rag_config.json` if present
3. **Environment variables** - Override both defaults and file settings

## Troubleshooting

### Common Configuration Issues

1. **Storage path not accessible**
   - Ensure the path exists and is writable
   - Check permissions for the application user

2. **Memory limit too restrictive**
   - Increase `RAG_MAX_MEMORY_MB` if processing fails
   - Monitor actual memory usage

3. **FIPS validation errors**
   - Ensure `RAG_HASH_ALGORITHM` is set to a FIPS-approved algorithm
   - Verify `OPENSSL_FIPS=1` is set correctly

4. **Demo limits reached**
   - Increase `RAG_DEMO_MAX_PAGES` or `RAG_DEMO_MAX_PDFS`
   - Or disable demo mode with `RAG_DEMO_MODE=false`

### Configuration Health Check

To verify your configuration is valid, you can run:

```python
from config import RAGConfigManager

manager = RAGConfigManager()
try:
    config = manager.load_config()
    print("✓ Configuration is valid")
except ValueError as e:
    print(f"✗ Configuration errors: {e}")
```

## Security Considerations

- Store sensitive configuration in environment variables, not in files
- Use FIPS-compliant settings in production environments
- Regularly review and audit configuration settings
- Limit memory and storage usage to prevent resource exhaustion
- Ensure storage paths are secured and not publicly accessible