# Chat Guide 06: Configuration & Settings

## Objective
Implement comprehensive configuration management for RAG functionality, including environment variables, configuration files, and runtime settings that support the demo environment and OpenShift deployment.

## Prerequisites
- Chat Guide 01-05 completed (full RAG system with UI)
- RAG pipeline and user interface working
- Document upload and management functional
- Feature branch: `feature/rag-implementation`

## Goals for This Chat Session
1. Add RAG configuration to existing config system
2. Create environment variables for RAG settings
3. Implement runtime configuration validation
4. Add configuration persistence and management
5. Create deployment-specific configuration templates

## Tasks to Complete

### 1. RAG Configuration Extension
**What to Ask For:**
```
Extend config.py to include RAG configuration:
- Add RAGConfig dataclass with all RAG settings
- Integrate with existing AppConfig structure
- Environment variable override support
- Configuration validation and defaults
- FIPS-compliant configuration options
```

### 2. Environment Variables
**What to Ask For:**
```
Define environment variables for RAG deployment:
- ENABLE_RAG=true/false
- RAG_STORAGE_PATH=/app/rag_storage
- RAG_MAX_MEMORY_MB=100
- RAG_CHUNK_SIZE=1000
- RAG_CHUNK_OVERLAP=200
- RAG_MAX_DOCUMENTS=100
- RAG_RELEVANCE_THRESHOLD=0.1
- Add validation and documentation
```

### 3. Configuration Validation
**What to Ask For:**
```
Implement configuration validation and error handling:
- Validate RAG storage path accessibility
- Check memory limit settings
- Validate document size and count limits
- Verify FIPS compliance settings
- Configuration health checks
```

### 4. Runtime Configuration Management
**What to Ask For:**
```
Add runtime configuration features:
- Configuration hot-reload without restart
- Configuration export/import functionality
- Configuration backup and restore
- Default configuration reset
- Configuration conflict resolution
```

### 5. Session State Configuration
**What to Ask For:**
```
Extend session state management for RAG:
- RAG-specific session state initialization
- Configuration persistence across sessions
- User preference storage
- Configuration migration handling
- Session cleanup for RAG data
```

### 6. Deployment Configuration Templates
**What to Ask For:**
```
Create deployment-specific configuration:
- Development environment defaults
- Demo environment configuration
- OpenShift production settings
- FIPS-compliant configuration profiles
- Configuration documentation and examples
```

### 7. Testing
**What to Ask For:**
```
Create tests/test_rag_config.py:
- Test configuration loading and validation
- Test environment variable override
- Test configuration validation errors
- Test runtime configuration changes
- Test deployment-specific configurations
```

## Expected Outputs After This Chat
- [ ] Extended `config.py` with comprehensive RAG configuration
- [ ] Environment variables defined and documented
- [ ] Configuration validation and error handling
- [ ] Runtime configuration management features
- [ ] Session state integration for RAG settings
- [ ] Deployment configuration templates
- [ ] Configuration tests and validation

## Key Implementation Details

### RAG Configuration Structure
```python
@dataclass
class RAGConfig:
    """RAG-specific configuration settings"""
    # Core RAG settings
    enabled: bool = True
    storage_path: str = "./rag_storage"
    max_memory_mb: int = 100
    
    # Document processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_document_size_mb: int = 10
    max_documents: int = 100
    supported_formats: List[str] = field(default_factory=lambda: ["pdf"])
    
    # Search settings
    max_search_results: int = 5
    relevance_threshold: float = 0.1
    tfidf_max_features: int = 5000
    
    # Performance settings
    cache_size: int = 50
    index_rebuild_threshold: int = 10
    
    # FIPS compliance
    fips_mode: bool = True
    hash_algorithm: str = "sha256"
    
    # Demo environment specific
    demo_mode: bool = False
    demo_max_pages: int = 300
    demo_max_pdfs: int = 75

@dataclass
class AppConfig:
    """Extended main application configuration"""
    # ... existing configuration ...
    
    # RAG configuration
    rag: RAGConfig = field(default_factory=RAGConfig)
```

### Environment Variable Integration
```python
def load_rag_config_from_env(config: RAGConfig) -> RAGConfig:
    """Load RAG configuration from environment variables"""
    
    # Core settings
    config.enabled = os.getenv("ENABLE_RAG", "true").lower() == "true"
    config.storage_path = os.getenv("RAG_STORAGE_PATH", config.storage_path)
    
    # Memory and performance
    if os.getenv("RAG_MAX_MEMORY_MB"):
        config.max_memory_mb = int(os.getenv("RAG_MAX_MEMORY_MB"))
    
    # Document processing
    if os.getenv("RAG_CHUNK_SIZE"):
        config.chunk_size = int(os.getenv("RAG_CHUNK_SIZE"))
    if os.getenv("RAG_CHUNK_OVERLAP"):
        config.chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP"))
    
    # Search settings
    if os.getenv("RAG_RELEVANCE_THRESHOLD"):
        config.relevance_threshold = float(os.getenv("RAG_RELEVANCE_THRESHOLD"))
    if os.getenv("RAG_MAX_SEARCH_RESULTS"):
        config.max_search_results = int(os.getenv("RAG_MAX_SEARCH_RESULTS"))
    
    # FIPS compliance
    config.fips_mode = os.getenv("OPENSSL_FIPS", "0") == "1"
    
    # Demo mode
    config.demo_mode = os.getenv("RAG_DEMO_MODE", "false").lower() == "true"
    
    return config
```

### Configuration Validation
```python
class RAGConfigValidator:
    """Validate RAG configuration settings"""
    
    def validate_config(self, config: RAGConfig) -> List[str]:
        """Validate configuration and return error messages"""
        errors = []
        
        # Storage path validation
        if not self._validate_storage_path(config.storage_path):
            errors.append(f"Invalid storage path: {config.storage_path}")
        
        # Memory limits
        if config.max_memory_mb < 50:
            errors.append("RAG memory limit too low (minimum 50MB)")
        if config.max_memory_mb > 1000:
            errors.append("RAG memory limit too high (maximum 1000MB)")
        
        # Chunk settings
        if config.chunk_size < 100:
            errors.append("Chunk size too small (minimum 100 characters)")
        if config.chunk_overlap >= config.chunk_size:
            errors.append("Chunk overlap must be less than chunk size")
        
        # Search settings
        if config.relevance_threshold < 0 or config.relevance_threshold > 1:
            errors.append("Relevance threshold must be between 0 and 1")
        
        # Demo mode validation
        if config.demo_mode:
            if config.demo_max_pages < 100:
                errors.append("Demo max pages too low (minimum 100)")
        
        # FIPS compliance
        if config.fips_mode and config.hash_algorithm != "sha256":
            errors.append("FIPS mode requires SHA-256 hash algorithm")
        
        return errors
    
    def _validate_storage_path(self, path: str) -> bool:
        """Validate storage path accessibility"""
        try:
            os.makedirs(path, exist_ok=True)
            test_file = os.path.join(path, ".test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            return True
        except Exception:
            return False
```

### Runtime Configuration Management
```python
class RAGConfigManager:
    """Manage RAG configuration at runtime"""
    
    def __init__(self, config_path: str = "rag_config.json"):
        self.config_path = config_path
        self.config = None
        self.validator = RAGConfigValidator()
    
    def load_config(self) -> RAGConfig:
        """Load configuration from file and environment"""
        # Load base config
        base_config = RAGConfig()
        
        # Override with file if exists
        if os.path.exists(self.config_path):
            base_config = self._load_from_file(base_config)
        
        # Override with environment variables
        final_config = load_rag_config_from_env(base_config)
        
        # Validate configuration
        errors = self.validator.validate_config(final_config)
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        
        self.config = final_config
        return final_config
    
    def save_config(self, config: RAGConfig) -> None:
        """Save configuration to file"""
        import json
        from dataclasses import asdict
        
        with open(self.config_path, "w") as f:
            json.dump(asdict(config), f, indent=2)
    
    def reset_to_defaults(self) -> RAGConfig:
        """Reset configuration to defaults"""
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        return self.load_config()
```

### Session State Integration
```python
def initialize_rag_session_state(config: RAGConfig):
    """Initialize RAG-specific session state"""
    # RAG configuration
    if "rag_config" not in st.session_state:
        st.session_state.rag_config = config
    
    # RAG state
    if "rag_enabled" not in st.session_state:
        st.session_state.rag_enabled = config.enabled
    
    if "rag_max_sources" not in st.session_state:
        st.session_state.rag_max_sources = config.max_search_results
    
    if "rag_threshold" not in st.session_state:
        st.session_state.rag_threshold = config.relevance_threshold
    
    # Document state
    if "uploaded_documents" not in st.session_state:
        st.session_state.uploaded_documents = []
    
    if "rag_storage_initialized" not in st.session_state:
        st.session_state.rag_storage_initialized = False
```

### Deployment Configuration Templates

#### Development Environment
```python
# config/dev_config.py
DEV_RAG_CONFIG = {
    "enabled": True,
    "storage_path": "./dev_rag_storage",
    "max_memory_mb": 200,
    "chunk_size": 800,
    "max_documents": 20,
    "demo_mode": True,
    "fips_mode": False  # Relaxed for development
}
```

#### Demo Environment
```python
# config/demo_config.py
DEMO_RAG_CONFIG = {
    "enabled": True,
    "storage_path": "/app/rag_storage",
    "max_memory_mb": 100,
    "chunk_size": 1000,
    "max_documents": 100,
    "demo_mode": True,
    "demo_max_pages": 300,
    "demo_max_pdfs": 75,
    "fips_mode": True
}
```

#### OpenShift Production
```python
# config/openshift_config.py
OPENSHIFT_RAG_CONFIG = {
    "enabled": True,
    "storage_path": "/mnt/rag-storage",
    "max_memory_mb": 100,
    "chunk_size": 1000,
    "max_documents": 200,
    "fips_mode": True,
    "hash_algorithm": "sha256",
    "cache_size": 100
}
```

## Configuration Documentation

### Environment Variables Reference
```bash
# Core RAG Settings
ENABLE_RAG=true                    # Enable/disable RAG functionality
RAG_STORAGE_PATH=/app/rag_storage  # Path for RAG data storage

# Memory and Performance
RAG_MAX_MEMORY_MB=100             # Maximum memory usage for RAG
RAG_CACHE_SIZE=50                 # Number of cached documents

# Document Processing
RAG_CHUNK_SIZE=1000               # Text chunk size in characters
RAG_CHUNK_OVERLAP=200             # Overlap between chunks
RAG_MAX_DOCUMENT_SIZE_MB=10       # Maximum individual document size
RAG_MAX_DOCUMENTS=100             # Maximum total documents

# Search Configuration
RAG_MAX_SEARCH_RESULTS=5          # Maximum search results per query
RAG_RELEVANCE_THRESHOLD=0.1       # Minimum relevance score
RAG_TFIDF_MAX_FEATURES=5000       # TF-IDF feature limit

# Demo Environment
RAG_DEMO_MODE=true                # Enable demo-specific limits
RAG_DEMO_MAX_PAGES=300            # Maximum pages in demo mode
RAG_DEMO_MAX_PDFS=75              # Maximum PDFs in demo mode

# FIPS Compliance
OPENSSL_FIPS=1                    # Enable FIPS mode
RAG_HASH_ALGORITHM=sha256         # Hash algorithm to use
```

## Success Criteria for This Chat
- ✅ RAG configuration integrates cleanly with existing config system
- ✅ Environment variables properly override configuration
- ✅ Configuration validation prevents invalid settings
- ✅ Runtime configuration management works smoothly
- ✅ Deployment templates support different environments
- ✅ All configuration changes are tested and documented

## Next Chat Guide
After completing configuration management, proceed to **Chat Guide 07: Testing & Validation** to create comprehensive tests for the entire RAG system.

## Notes for Implementation
- Maintain backward compatibility with existing configuration
- Ensure FIPS compliance is properly configured
- Provide clear error messages for configuration issues
- Document all configuration options thoroughly
- Test configuration in various deployment scenarios
- Consider configuration security (no secrets in logs)