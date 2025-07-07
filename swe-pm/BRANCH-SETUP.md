# Feature Branch Setup for RAG Implementation

## Create and Setup Feature Branch

Run these commands to set up your feature branch for RAG implementation:

```bash
# Ensure you're on the latest feature/api-first-refactor branch
git checkout feature/api-first-refactor
git pull origin feature/api-first-refactor

# Create new feature branch for RAG
git checkout -b feature/rag-implementation
git push -u origin feature/rag-implementation
```

## Verify Branch Setup

```bash
# Check current branch
git branch -v

# Verify remote tracking
git status

# Should show:
# On branch feature/rag-implementation
# Your branch is up to date with 'origin/feature/rag-implementation'.
```

## Project Status Check

Before starting RAG implementation, verify the current project state:

```bash
# Check if app runs
streamlit run app.py

# Test API client functionality
# - Try connecting to an API endpoint
# - Test chat functionality
# - Test image analysis if available

# Run existing tests
python -m pytest tests/ -v
```

## Ready to Start

Once your branch is set up and the current functionality is verified:

1. **Start with Chat Guide 01**: [Document Processing Foundation](01-DOCUMENT-PROCESSING-FOUNDATION.md)
2. **Work sequentially**: Complete each guide before moving to the next
3. **Test incrementally**: Validate functionality after each guide
4. **Commit regularly**: Make commits after completing each major component

## Commit Strategy

Suggested commit pattern for each chat guide:

```bash
# After completing Chat Guide 01
git add rag/ tests/test_document_processing.py requirements.txt
git commit -m "feat: add document processing foundation with Docling

- Add DoclingProcessor class for PDF text extraction
- Implement FIPS-compliant text chunking with metadata
- Add comprehensive document processing tests
- Verify FIPS compliance for all operations

Closes: Chat Guide 01"

# Push progress regularly
git push origin feature/rag-implementation
```

## Backup Strategy

Before starting major changes:

```bash
# Create a backup tag
git tag -a backup-before-rag -m "Backup before RAG implementation"
git push origin backup-before-rag
```

This allows easy rollback if needed:

```bash
# If you need to rollback (only if major issues)
git reset --hard backup-before-rag
```

## Branch Merge Strategy

After completing all 8 chat guides:

1. **Final testing**: Run full test suite
2. **Create pull request**: From `feature/rag-implementation` to `feature/api-first-refactor`
3. **Review and test**: Deploy to test environment
4. **Merge**: After successful validation

Example final PR description:
```markdown
# Add File-Based RAG Functionality

## Summary
Implements comprehensive RAG functionality using file-based storage to avoid FIPS compliance issues with vector databases.

## Features Added
- Docling-based PDF text extraction and chunking
- File-based document storage with memory management  
- TF-IDF similarity search with hybrid scoring
- End-to-end RAG pipeline with source citations
- Document upload and management UI
- RAG-enhanced chat interface
- FIPS-compliant throughout

## Testing
- All new components have comprehensive unit tests
- Integration tests validate end-to-end workflows
- FIPS compliance validated in FIPS-enabled environment
- Performance tested with demo requirements (75 PDFs, 300 pages)

## Deployment
- OpenShift manifests updated for persistent storage
- Configuration management for RAG settings
- Production deployment documentation included
```

You're all set! Start with [Chat Guide 01](01-DOCUMENT-PROCESSING-FOUNDATION.md) when ready.