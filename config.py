"""Configuration management for the FIPS Chat application."""

import os
import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict


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
    """Main application configuration."""
    # API settings
    default_api_endpoint: Optional[str] = None
    default_api_key: Optional[str] = None
    default_api_provider: str = "openai_compatible"
    request_timeout: int = 60
    
    # File handling
    max_file_size_mb: int = 10
    supported_image_formats: List[str] = field(default_factory=lambda: ["jpg", "jpeg", "png", "webp", "gif"])
    
    # Model generation parameters
    temperature: float = 0.7
    max_tokens: int = 2048
    
    # UI settings
    theme: str = "light"
    chat_history_limit: int = 50
    
    # Model discovery settings
    model_cache_ttl: int = 300  # 5 minutes
    auto_discover_models: bool = True
    
    # RAG configuration
    rag: RAGConfig = field(default_factory=RAGConfig)


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
        if config.chunk_overlap < 0:
            errors.append("Chunk overlap cannot be negative")
        
        # Document limits
        if config.max_document_size_mb < 1:
            errors.append("Maximum document size too small (minimum 1MB)")
        if config.max_document_size_mb > 100:
            errors.append("Maximum document size too large (maximum 100MB)")
        if config.max_documents < 1:
            errors.append("Maximum documents must be at least 1")
        
        # Search settings
        if config.relevance_threshold < 0 or config.relevance_threshold > 1:
            errors.append("Relevance threshold must be between 0 and 1")
        if config.max_search_results < 1:
            errors.append("Maximum search results must be at least 1")
        if config.max_search_results > 20:
            errors.append("Maximum search results too high (maximum 20)")
        
        # Performance settings
        if config.cache_size < 10:
            errors.append("Cache size too small (minimum 10)")
        if config.tfidf_max_features < 100:
            errors.append("TF-IDF max features too small (minimum 100)")
        
        # Demo mode validation
        if config.demo_mode:
            if config.demo_max_pages < 100:
                errors.append("Demo max pages too low (minimum 100)")
            if config.demo_max_pdfs < 10:
                errors.append("Demo max PDFs too low (minimum 10)")
        
        # FIPS compliance
        if config.fips_mode and config.hash_algorithm not in ["sha256", "sha384", "sha512"]:
            errors.append("FIPS mode requires SHA-256, SHA-384, or SHA-512 hash algorithm")
        
        # Supported formats validation
        if not config.supported_formats:
            errors.append("At least one supported format must be specified")
        
        return errors
    
    def _validate_storage_path(self, path: str) -> bool:
        """Validate storage path accessibility"""
        if not path or not isinstance(path, str):
            return False
        
        try:
            # Expand user path and make absolute
            expanded_path = os.path.abspath(os.path.expanduser(path))
            
            # Try to create directory if it doesn't exist
            os.makedirs(expanded_path, exist_ok=True)
            
            # Test write access
            test_file = os.path.join(expanded_path, ".test_access")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            return True
        except Exception:
            return False


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
        final_config = self._load_from_env(base_config)
        
        # Validate configuration
        errors = self.validator.validate_config(final_config)
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        
        self.config = final_config
        return final_config
    
    def _load_from_file(self, base_config: RAGConfig) -> RAGConfig:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, "r") as f:
                file_config = json.load(f)
            
            # Update base config with file values
            config_dict = asdict(base_config)
            config_dict.update(file_config)
            
            # Convert back to RAGConfig
            return RAGConfig(**config_dict)
        except Exception as e:
            print(f"Warning: Failed to load config from {self.config_path}: {e}")
            return base_config
    
    def _load_from_env(self, config: RAGConfig) -> RAGConfig:
        """Load RAG configuration from environment variables"""
        
        # Core settings
        enable_rag = os.getenv("ENABLE_RAG")
        if enable_rag:
            config.enabled = enable_rag.lower() == "true"
        
        storage_path = os.getenv("RAG_STORAGE_PATH")
        if storage_path:
            config.storage_path = storage_path
        
        # Memory and performance
        max_memory = os.getenv("RAG_MAX_MEMORY_MB")
        if max_memory:
            try:
                config.max_memory_mb = int(max_memory)
            except ValueError:
                print("Warning: Invalid RAG_MAX_MEMORY_MB value")
        
        cache_size = os.getenv("RAG_CACHE_SIZE")
        if cache_size:
            try:
                config.cache_size = int(cache_size)
            except ValueError:
                print("Warning: Invalid RAG_CACHE_SIZE value")
        
        # Document processing
        chunk_size = os.getenv("RAG_CHUNK_SIZE")
        if chunk_size:
            try:
                config.chunk_size = int(chunk_size)
            except ValueError:
                print("Warning: Invalid RAG_CHUNK_SIZE value")
        
        chunk_overlap = os.getenv("RAG_CHUNK_OVERLAP")
        if chunk_overlap:
            try:
                config.chunk_overlap = int(chunk_overlap)
            except ValueError:
                print("Warning: Invalid RAG_CHUNK_OVERLAP value")
        
        max_doc_size = os.getenv("RAG_MAX_DOCUMENT_SIZE_MB")
        if max_doc_size:
            try:
                config.max_document_size_mb = int(max_doc_size)
            except ValueError:
                print("Warning: Invalid RAG_MAX_DOCUMENT_SIZE_MB value")
        
        max_docs = os.getenv("RAG_MAX_DOCUMENTS")
        if max_docs:
            try:
                config.max_documents = int(max_docs)
            except ValueError:
                print("Warning: Invalid RAG_MAX_DOCUMENTS value")
        
        # Search settings
        relevance_threshold = os.getenv("RAG_RELEVANCE_THRESHOLD")
        if relevance_threshold:
            try:
                config.relevance_threshold = float(relevance_threshold)
            except ValueError:
                print("Warning: Invalid RAG_RELEVANCE_THRESHOLD value")
        
        max_search_results = os.getenv("RAG_MAX_SEARCH_RESULTS")
        if max_search_results:
            try:
                config.max_search_results = int(max_search_results)
            except ValueError:
                print("Warning: Invalid RAG_MAX_SEARCH_RESULTS value")
        
        tfidf_features = os.getenv("RAG_TFIDF_MAX_FEATURES")
        if tfidf_features:
            try:
                config.tfidf_max_features = int(tfidf_features)
            except ValueError:
                print("Warning: Invalid RAG_TFIDF_MAX_FEATURES value")
        
        # FIPS compliance
        fips_env = os.getenv("OPENSSL_FIPS")
        if fips_env:
            config.fips_mode = fips_env == "1"
        
        hash_algo = os.getenv("RAG_HASH_ALGORITHM")
        if hash_algo:
            config.hash_algorithm = hash_algo
        
        # Demo mode
        demo_mode = os.getenv("RAG_DEMO_MODE")
        if demo_mode:
            config.demo_mode = demo_mode.lower() == "true"
        
        demo_max_pages = os.getenv("RAG_DEMO_MAX_PAGES")
        if demo_max_pages:
            try:
                config.demo_max_pages = int(demo_max_pages)
            except ValueError:
                print("Warning: Invalid RAG_DEMO_MAX_PAGES value")
        
        demo_max_pdfs = os.getenv("RAG_DEMO_MAX_PDFS")
        if demo_max_pdfs:
            try:
                config.demo_max_pdfs = int(demo_max_pdfs)
            except ValueError:
                print("Warning: Invalid RAG_DEMO_MAX_PDFS value")
        
        return config
    
    def save_config(self, config: RAGConfig) -> None:
        """Save configuration to file"""
        try:
            with open(self.config_path, "w") as f:
                json.dump(asdict(config), f, indent=2)
        except Exception as e:
            raise RuntimeError(f"Failed to save configuration: {e}")
    
    def reset_to_defaults(self) -> RAGConfig:
        """Reset configuration to defaults"""
        if os.path.exists(self.config_path):
            try:
                os.remove(self.config_path)
            except Exception as e:
                print(f"Warning: Could not remove config file: {e}")
        return self.load_config()
    
    def export_config(self, export_path: str) -> None:
        """Export current configuration to file"""
        if not self.config:
            raise RuntimeError("No configuration loaded")
        
        with open(export_path, "w") as f:
            json.dump(asdict(self.config), f, indent=2)
    
    def import_config(self, import_path: str) -> RAGConfig:
        """Import configuration from file"""
        if not os.path.exists(import_path):
            raise FileNotFoundError(f"Configuration file not found: {import_path}")
        
        # Backup current config if it exists
        if os.path.exists(self.config_path):
            backup_path = f"{self.config_path}.backup"
            try:
                os.rename(self.config_path, backup_path)
            except Exception as e:
                print(f"Warning: Could not create backup: {e}")
        
        # Copy import file to config path
        try:
            import shutil
            shutil.copy2(import_path, self.config_path)
            return self.load_config()
        except Exception as e:
            # Restore backup if copy failed
            backup_path = f"{self.config_path}.backup"
            if os.path.exists(backup_path):
                os.rename(backup_path, self.config_path)
            raise RuntimeError(f"Failed to import configuration: {e}")


# Deployment-specific configuration templates
DEPLOYMENT_CONFIGS = {
    "development": {
        "enabled": True,
        "storage_path": "./dev_rag_storage",
        "max_memory_mb": 200,
        "chunk_size": 800,
        "max_documents": 20,
        "demo_mode": True,
        "fips_mode": False,  # Relaxed for development
        "cache_size": 25
    },
    "demo": {
        "enabled": True,
        "storage_path": "/app/rag_storage",
        "max_memory_mb": 100,
        "chunk_size": 1000,
        "max_documents": 100,
        "demo_mode": True,
        "demo_max_pages": 300,
        "demo_max_pdfs": 75,
        "fips_mode": True,
        "cache_size": 50
    },
    "openshift": {
        "enabled": True,
        "storage_path": "/mnt/rag-storage",
        "max_memory_mb": 100,
        "chunk_size": 1000,
        "max_documents": 200,
        "fips_mode": True,
        "hash_algorithm": "sha256",
        "cache_size": 100,
        "demo_mode": False
    }
}


def load_rag_config_from_env(config: RAGConfig) -> RAGConfig:
    """Load RAG configuration from environment variables (legacy function for compatibility)"""
    manager = RAGConfigManager()
    return manager._load_from_env(config)


def get_deployment_config(deployment_type: str) -> Dict[str, Any]:
    """Get deployment-specific configuration template"""
    return DEPLOYMENT_CONFIGS.get(deployment_type, DEPLOYMENT_CONFIGS["demo"]).copy()


def apply_deployment_config(config: RAGConfig, deployment_type: str) -> RAGConfig:
    """Apply deployment-specific configuration to RAGConfig"""
    template = get_deployment_config(deployment_type)
    
    # Update config with template values
    config_dict = asdict(config)
    config_dict.update(template)
    
    return RAGConfig(**config_dict)


# API endpoint templates for common providers
API_TEMPLATES = {
    "openai": {
        "name": "OpenAI",
        "endpoint": "https://api.openai.com/v1",
        "requires_key": True,
        "example_models": ["gpt-4", "gpt-3.5-turbo", "gpt-4-vision-preview"]
    },
    "anthropic": {
        "name": "Anthropic Claude",
        "endpoint": "https://api.anthropic.com",
        "requires_key": True,
        "example_models": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"]
    },
    "vllm": {
        "name": "vLLM Server",
        "endpoint": "http://localhost:8000/v1",
        "requires_key": False,
        "example_models": ["Auto-discovered from endpoint"]
    },
    "ollama": {
        "name": "Ollama API",
        "endpoint": "http://localhost:11434/v1",
        "requires_key": False,
        "example_models": ["Auto-discovered from endpoint"]
    },
    "text-generation-webui": {
        "name": "Text Generation WebUI",
        "endpoint": "http://localhost:5000/v1",
        "requires_key": False,
        "example_models": ["Auto-discovered from endpoint"]
    }
}


def get_config() -> AppConfig:
    """Get application configuration, allowing environment variable overrides."""
    config = AppConfig()
    
    # Override with environment variables if present
    config.default_api_endpoint = os.getenv("API_ENDPOINT", config.default_api_endpoint)
    config.default_api_key = os.getenv("API_KEY", config.default_api_key)
    config.default_api_provider = os.getenv("API_PROVIDER", config.default_api_provider)
    
    request_timeout = os.getenv("REQUEST_TIMEOUT")
    if request_timeout:
        try:
            config.request_timeout = int(request_timeout)
        except ValueError:
            print("Warning: Invalid REQUEST_TIMEOUT value")
    
    max_file_size = os.getenv("MAX_FILE_SIZE_MB")
    if max_file_size:
        try:
            config.max_file_size_mb = int(max_file_size)
        except ValueError:
            print("Warning: Invalid MAX_FILE_SIZE_MB value")
    
    temperature = os.getenv("TEMPERATURE")
    if temperature:
        try:
            config.temperature = float(temperature)
        except ValueError:
            print("Warning: Invalid TEMPERATURE value")
    
    max_tokens = os.getenv("MAX_TOKENS")
    if max_tokens:
        try:
            config.max_tokens = int(max_tokens)
        except ValueError:
            print("Warning: Invalid MAX_TOKENS value")
    
    model_cache_ttl = os.getenv("MODEL_CACHE_TTL")
    if model_cache_ttl:
        try:
            config.model_cache_ttl = int(model_cache_ttl)
        except ValueError:
            print("Warning: Invalid MODEL_CACHE_TTL value")
    
    auto_discover = os.getenv("AUTO_DISCOVER_MODELS")
    if auto_discover:
        config.auto_discover_models = auto_discover.lower() in ("true", "1", "yes")
    
    return config


def get_api_template(template_name: str) -> Optional[Dict[str, Any]]:
    """Get API template configuration."""
    return API_TEMPLATES.get(template_name)


def get_all_api_templates() -> Dict[str, Dict[str, Any]]:
    """Get all available API templates."""
    return API_TEMPLATES.copy()


def is_vision_capable_model(model_name: str) -> bool:
    """Check if a model likely supports vision/image analysis based on name."""
    name_lower = model_name.lower()
    
    # Common vision model indicators
    vision_indicators = [
        "vision", "llava", "claude-3", "gpt-4-vision", "gpt-4o", 
        "gpt-4-turbo", "gemini-pro-vision", "pixtral", "qwen-vl"
    ]
    
    return any(indicator in name_lower for indicator in vision_indicators)


def is_code_capable_model(model_name: str) -> bool:
    """Check if a model is optimized for code generation."""
    name_lower = model_name.lower()
    
    # Common code model indicators
    code_indicators = [
        "code", "coder", "codellama", "starcoder", "deepseek-coder",
        "wizard-coder", "magicoder", "granite-code"
    ]
    
    return any(indicator in name_lower for indicator in code_indicators)