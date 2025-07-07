# RAG System Testing Infrastructure

This directory contains comprehensive tests for the RAG (Retrieval-Augmented Generation) system, implementing all requirements from Chat Guide 07: Testing & Validation.

## Test Categories

### 1. Unit Tests
**Files:** `test_document_processing.py`, `test_file_storage.py`, `test_search_engine.py`, `test_rag_pipeline.py`, `test_rag_config.py`

- **Purpose:** Test individual components in isolation
- **Coverage:** Document processing, file storage, search engine, RAG pipeline, configuration
- **Features:** Mocked dependencies, error condition testing, FIPS compliance validation

### 2. Integration Tests
**Files:** `test_rag_integration.py`

- **Purpose:** End-to-end workflow testing
- **Coverage:** Document upload through response generation
- **Features:** Multiple document scenarios, error recovery, memory constraints

### 3. Performance Tests
**Files:** `test_rag_performance.py`

- **Purpose:** Memory usage, response times, concurrent access
- **Coverage:** Memory limits (100MB), response times (<5s), concurrent users
- **Features:** Memory leak detection, performance benchmarks, large document sets

### 4. FIPS Compliance Tests
**Files:** `test_fips_compliance.py`

- **Purpose:** Cryptographic compliance validation
- **Coverage:** Hash algorithms, MD5 prohibition, security standards
- **Features:** Environment variable detection, production mode validation

### 5. Demo Scenario Tests
**Files:** `test_demo_scenario.py`

- **Purpose:** Demo-specific requirements validation
- **Coverage:** 75 PDFs, 300 pages, memory constraints
- **Features:** Exact demo limits, quality metrics, performance validation

### 6. Application Tests
**Files:** `test_app_integration.py`, `test_session_manager.py`, `test_image_processing.py`, `test_ollama_client.py`, `test_config.py`

- **Purpose:** Application-level functionality
- **Coverage:** Session management, image processing, client integration

## Test Infrastructure

### Fixtures and Test Data
**Directory:** `tests/fixtures/`

- **Test Data Generator:** `generate_test_data.py` - Creates sample PDFs and mock data
- **PDF Generation:** Uses ReportLab to create realistic test documents
- **Demo Data Set:** Generates exactly 75 PDFs with 300 total pages
- **Query Templates:** Predefined test queries for validation

### Test Runner
**File:** `run_all_tests.py`

Comprehensive test runner with:
- Category-based test execution
- Parallel test support
- Performance monitoring  
- Coverage reporting
- Detailed test reports

## Usage

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
python tests/run_all_tests.py

# Run specific category
python tests/run_all_tests.py --categories unit

# Quick test (unit tests only)
python tests/run_all_tests.py --quick

# Verbose output with coverage
python tests/run_all_tests.py --verbose --coverage
```

### Individual Test Categories

```bash
# Unit tests only
python tests/run_all_tests.py -c unit

# Performance tests
python tests/run_all_tests.py -c performance

# FIPS compliance
python tests/run_all_tests.py -c fips

# Demo scenario validation
python tests/run_all_tests.py -c demo

# Integration tests
python tests/run_all_tests.py -c integration
```

### Using pytest Directly

```bash
# Run specific test file
pytest tests/test_document_processing.py -v

# Run with coverage
pytest tests/ --cov=rag --cov-report=html

# Run performance tests with timeout
pytest tests/test_rag_performance.py --timeout=900

# Run in parallel
pytest tests/ -n auto
```

## Test Configuration

### Environment Variables

- `RAG_TEST_MODE=1` - Enables test mode
- `RAG_DISABLE_EXTERNAL_DEPS=1` - Disables external dependencies
- `RAG_TEST_COVERAGE=1` - Enables coverage reporting
- `OPENSSL_FIPS=1` - Enables FIPS mode for compliance testing

### Memory Constraints

Tests are designed to validate the 100MB memory limit:
- Memory monitoring during test execution
- Memory leak detection
- Peak memory usage validation

### Performance Requirements

Tests validate the following performance criteria:
- Query response time: <5 seconds
- Index build time: <30 seconds (large document sets)
- Memory usage: <100MB increase
- Concurrent user support: 5+ users

## Test Data

### Generated Test Documents

The test suite automatically generates:
- **Demo Document Set:** 75 PDFs, 300 total pages
- **Topic Variety:** Machine learning, data science, security
- **Realistic Content:** Multi-page documents with structured content
- **Metadata:** Complete document metadata for validation

### Test Queries

Predefined queries for different difficulty levels:
- **Basic:** Simple concept queries
- **Intermediate:** Multi-topic analysis
- **Advanced:** Complex reasoning and synthesis

## FIPS Compliance Testing

### Hash Algorithm Validation
- SHA-256, SHA-384, SHA-512 (approved)
- MD5 prohibition verification
- Production environment simulation

### Cryptographic Operations
- Document hashing validation
- File integrity checking
- Security standard compliance

## Performance Monitoring

### Memory Tracking
- Continuous memory sampling
- Peak usage detection
- Memory leak identification
- LRU cache validation

### Response Time Analysis
- Query processing benchmarks
- Index build performance
- Concurrent access testing
- Performance regression detection

## Success Criteria

### Unit Tests
- ✅ >90% code coverage
- ✅ All components tested in isolation
- ✅ Error conditions handled properly
- ✅ FIPS compliance validated

### Integration Tests
- ✅ End-to-end workflows functional
- ✅ Multiple document scenarios
- ✅ Error recovery mechanisms
- ✅ Memory constraints respected

### Performance Tests
- ✅ Memory usage <100MB
- ✅ Query response time <5s
- ✅ Concurrent user support
- ✅ Large document set handling

### FIPS Compliance
- ✅ Only approved algorithms used
- ✅ MD5 usage prohibited
- ✅ Production mode validation
- ✅ Environment detection working

### Demo Scenario
- ✅ Exactly 75 PDFs supported
- ✅ 300 total pages processed
- ✅ Memory limits respected
- ✅ Performance requirements met

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test Timeouts**
   - Increase timeout values in test configuration
   - Check system resources during test execution

3. **Memory Test Failures**
   - Ensure sufficient system memory available
   - Close other applications during testing

4. **FIPS Test Failures**
   - Verify FIPS-enabled OpenSSL installation
   - Check environment variable settings

### Test Debugging

1. **Verbose Output**
   ```bash
   python tests/run_all_tests.py --verbose
   ```

2. **Individual Test Debugging**
   ```bash
   pytest tests/test_specific.py::TestClass::test_method -v -s
   ```

3. **Coverage Analysis**
   ```bash
   pytest tests/ --cov=rag --cov-report=html
   open htmlcov/index.html
   ```

## Contributing

### Adding New Tests

1. Follow existing test patterns
2. Use appropriate fixtures and mocks
3. Include both success and failure scenarios
4. Add performance considerations
5. Update test categories in `run_all_tests.py`

### Test Guidelines

- **Isolation:** Tests should not depend on external services
- **Determinism:** Tests should produce consistent results
- **Performance:** Tests should complete within reasonable time
- **Documentation:** Include clear docstrings and comments

## Integration with CI/CD

The test suite is designed for automated execution:

```yaml
# Example GitHub Actions workflow
- name: Run RAG Tests
  run: |
    pip install -r requirements.txt
    python tests/run_all_tests.py --coverage --output test_report.txt
- name: Upload Test Report
  uses: actions/upload-artifact@v3
  with:
    name: test-report
    path: test_report.txt
```

## Maintenance

### Regular Tasks

1. **Update test data** as system evolves
2. **Adjust performance thresholds** based on infrastructure
3. **Add new test scenarios** for new features
4. **Review and update FIPS compliance** requirements

### Performance Monitoring

- Monitor test execution times
- Track memory usage trends
- Identify performance regressions
- Update benchmarks as needed 