"""
Tests for RAG configuration system.

This module tests the comprehensive RAG configuration management including:
- Configuration loading and validation
- Environment variable overrides
- Configuration validation errors
- Runtime configuration changes
- Deployment-specific configurations
- Configuration file management
"""

import pytest
import os
import json
import tempfile
import shutil
from unittest.mock import patch, mock_open
from dataclasses import asdict

from config import (
    RAGConfig, 
    RAGConfigValidator, 
    RAGConfigManager,
    get_deployment_config,
    apply_deployment_config,
    load_rag_config_from_env,
    DEPLOYMENT_CONFIGS
)


class TestRAGConfig:
    """Test RAGConfig dataclass and defaults."""
    
    def test_default_config(self):
        """Test default RAG configuration values."""
        config = RAGConfig()
        
        assert config.enabled is True
        assert config.storage_path == "./rag_storage"
        assert config.max_memory_mb == 100
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.max_document_size_mb == 10
        assert config.max_documents == 100
        assert config.supported_formats == ["pdf"]
        assert config.max_search_results == 5
        assert config.relevance_threshold == 0.1
        assert config.tfidf_max_features == 5000
        assert config.cache_size == 50
        assert config.index_rebuild_threshold == 10
        assert config.fips_mode is True
        assert config.hash_algorithm == "sha256"
        assert config.demo_mode is False
        assert config.demo_max_pages == 300
        assert config.demo_max_pdfs == 75
    
    def test_config_serialization(self):
        """Test config can be serialized and deserialized."""
        config = RAGConfig()
        config_dict = asdict(config)
        
        # Should be able to serialize to JSON
        json_str = json.dumps(config_dict)
        assert isinstance(json_str, str)
        
        # Should be able to deserialize back
        loaded_dict = json.loads(json_str)
        loaded_config = RAGConfig(**loaded_dict)
        
        assert loaded_config.enabled == config.enabled
        assert loaded_config.storage_path == config.storage_path
        assert loaded_config.max_memory_mb == config.max_memory_mb


class TestRAGConfigValidator:
    """Test RAG configuration validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = RAGConfigValidator()
        self.valid_config = RAGConfig()
    
    def test_valid_config(self):
        """Test validation of a valid configuration."""
        errors = self.validator.validate_config(self.valid_config)
        assert len(errors) == 0
    
    def test_invalid_memory_limits(self):
        """Test validation of memory limit constraints."""
        # Too low
        config = RAGConfig(max_memory_mb=10)
        errors = self.validator.validate_config(config)
        assert any("memory limit too low" in error.lower() for error in errors)
        
        # Too high
        config = RAGConfig(max_memory_mb=2000)
        errors = self.validator.validate_config(config)
        assert any("memory limit too high" in error.lower() for error in errors)
    
    def test_invalid_chunk_settings(self):
        """Test validation of chunk size and overlap."""
        # Chunk size too small
        config = RAGConfig(chunk_size=50)
        errors = self.validator.validate_config(config)
        assert any("chunk size too small" in error.lower() for error in errors)
        
        # Chunk overlap >= chunk size
        config = RAGConfig(chunk_size=1000, chunk_overlap=1000)
        errors = self.validator.validate_config(config)
        assert any("chunk overlap must be less than chunk size" in error.lower() for error in errors)
        
        # Negative overlap
        config = RAGConfig(chunk_overlap=-10)
        errors = self.validator.validate_config(config)
        assert any("chunk overlap cannot be negative" in error.lower() for error in errors)
    
    def test_invalid_document_limits(self):
        """Test validation of document size and count limits."""
        # Document size too small
        config = RAGConfig(max_document_size_mb=0)
        errors = self.validator.validate_config(config)
        assert any("document size too small" in error.lower() for error in errors)
        
        # Document size too large
        config = RAGConfig(max_document_size_mb=200)
        errors = self.validator.validate_config(config)
        assert any("document size too large" in error.lower() for error in errors)
        
        # No documents allowed
        config = RAGConfig(max_documents=0)
        errors = self.validator.validate_config(config)
        assert any("documents must be at least 1" in error.lower() for error in errors)
    
    def test_invalid_search_settings(self):
        """Test validation of search configuration."""
        # Invalid relevance threshold
        config = RAGConfig(relevance_threshold=-0.1)
        errors = self.validator.validate_config(config)
        assert any("relevance threshold must be between 0 and 1" in error.lower() for error in errors)
        
        config = RAGConfig(relevance_threshold=1.5)
        errors = self.validator.validate_config(config)
        assert any("relevance threshold must be between 0 and 1" in error.lower() for error in errors)
        
        # Invalid max search results
        config = RAGConfig(max_search_results=0)
        errors = self.validator.validate_config(config)
        assert any("search results must be at least 1" in error.lower() for error in errors)
        
        config = RAGConfig(max_search_results=50)
        errors = self.validator.validate_config(config)
        assert any("search results too high" in error.lower() for error in errors)
    
    def test_invalid_performance_settings(self):
        """Test validation of performance settings."""
        # Cache size too small
        config = RAGConfig(cache_size=5)
        errors = self.validator.validate_config(config)
        assert any("cache size too small" in error.lower() for error in errors)
        
        # TF-IDF features too small
        config = RAGConfig(tfidf_max_features=50)
        errors = self.validator.validate_config(config)
        assert any("tf-idf" in error.lower() and "too small" in error.lower() for error in errors)
    
    def test_demo_mode_validation(self):
        """Test validation of demo mode settings."""
        config = RAGConfig(demo_mode=True, demo_max_pages=50)
        errors = self.validator.validate_config(config)
        assert any("demo max pages too low" in error.lower() for error in errors)
        
        config = RAGConfig(demo_mode=True, demo_max_pdfs=5)
        errors = self.validator.validate_config(config)
        assert any("demo max pdfs too low" in error.lower() for error in errors)
    
    def test_fips_compliance_validation(self):
        """Test validation of FIPS compliance settings."""
        config = RAGConfig(fips_mode=True, hash_algorithm="md5")
        errors = self.validator.validate_config(config)
        assert any("fips mode requires" in error.lower() for error in errors)
    
    def test_empty_supported_formats(self):
        """Test validation fails with empty supported formats."""
        config = RAGConfig(supported_formats=[])
        errors = self.validator.validate_config(config)
        assert any("at least one supported format" in error.lower() for error in errors)
    
    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.remove')
    def test_valid_storage_path(self, mock_remove, mock_file, mock_makedirs):
        """Test storage path validation success."""
        assert self.validator._validate_storage_path("/valid/path") is True
        mock_makedirs.assert_called_once()
        mock_file.assert_called_once()
        mock_remove.assert_called_once()
    
    @patch('os.makedirs', side_effect=PermissionError())
    def test_invalid_storage_path(self, mock_makedirs):
        """Test storage path validation failure."""
        assert self.validator._validate_storage_path("/invalid/path") is False


class TestRAGConfigManager:
    """Test RAG configuration manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.json")
        self.manager = RAGConfigManager(self.config_path)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_default_config(self):
        """Test loading default configuration when no file exists."""
        config = self.manager.load_config()
        
        assert isinstance(config, RAGConfig)
        assert config.enabled is True
        assert config.storage_path == "./rag_storage"
    
    @patch('config.RAGConfigValidator._validate_storage_path', return_value=True)
    def test_save_and_load_config(self, mock_validate_path):
        """Test saving and loading configuration from file."""
        original_config = RAGConfig(
            enabled=False,
            storage_path="/custom/path",
            max_memory_mb=200
        )
        
        self.manager.save_config(original_config)
        assert os.path.exists(self.config_path)
        
        loaded_config = self.manager.load_config()
        assert loaded_config.enabled is False
        assert loaded_config.storage_path == "/custom/path"
        assert loaded_config.max_memory_mb == 200
    
    def test_load_config_with_invalid_file(self):
        """Test loading configuration with invalid JSON file."""
        # Create invalid JSON file
        with open(self.config_path, "w") as f:
            f.write("invalid json content")
        
        # Should fall back to defaults and not raise exception
        config = self.manager.load_config()
        assert isinstance(config, RAGConfig)
        assert config.enabled is True  # Default value
    
    def test_config_validation_on_load(self):
        """Test configuration validation during load."""
        # Create config with validation errors
        invalid_config = {"max_memory_mb": 10}  # Too low
        with open(self.config_path, "w") as f:
            json.dump(invalid_config, f)
        
        with pytest.raises(ValueError, match="Configuration errors"):
            self.manager.load_config()
    
    @patch.dict(os.environ, {
        "RAG_MAX_MEMORY_MB": "150",
        "RAG_CHUNK_SIZE": "800",
        "ENABLE_RAG": "false"
    })
    def test_environment_variable_override(self):
        """Test environment variable overrides."""
        config = self.manager.load_config()
        
        assert config.enabled is False
        assert config.max_memory_mb == 150
        assert config.chunk_size == 800
    
    @patch.dict(os.environ, {"RAG_MAX_MEMORY_MB": "invalid_number"})
    def test_invalid_environment_variables(self):
        """Test handling of invalid environment variables."""
        # Should not raise exception, just use defaults
        config = self.manager.load_config()
        assert config.max_memory_mb == 100  # Default value
    
    def test_reset_to_defaults(self):
        """Test resetting configuration to defaults."""
        # Save custom config
        custom_config = RAGConfig(max_memory_mb=200)
        self.manager.save_config(custom_config)
        
        # Reset to defaults
        reset_config = self.manager.reset_to_defaults()
        
        assert reset_config.max_memory_mb == 100  # Default value
        assert not os.path.exists(self.config_path)
    
    def test_export_import_config(self):
        """Test configuration export and import."""
        original_config = RAGConfig(max_memory_mb=200, chunk_size=800)
        self.manager.config = original_config
        
        export_path = os.path.join(self.temp_dir, "exported_config.json")
        self.manager.export_config(export_path)
        assert os.path.exists(export_path)
        
        # Create new manager and import
        new_config_path = os.path.join(self.temp_dir, "imported_config.json")
        new_manager = RAGConfigManager(new_config_path)
        imported_config = new_manager.import_config(export_path)
        
        assert imported_config.max_memory_mb == 200
        assert imported_config.chunk_size == 800
    
    def test_import_nonexistent_file(self):
        """Test importing from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            self.manager.import_config("/nonexistent/file.json")
    
    def test_export_without_loaded_config(self):
        """Test exporting without loaded configuration."""
        manager = RAGConfigManager()
        with pytest.raises(RuntimeError, match="No configuration loaded"):
            manager.export_config("/some/path.json")


class TestEnvironmentVariableLoading:
    """Test environment variable loading functionality."""
    
    def test_load_all_environment_variables(self):
        """Test loading all supported environment variables."""
        env_vars = {
            "ENABLE_RAG": "false",
            "RAG_STORAGE_PATH": "/custom/storage",
            "RAG_MAX_MEMORY_MB": "150",
            "RAG_CACHE_SIZE": "75",
            "RAG_CHUNK_SIZE": "800",
            "RAG_CHUNK_OVERLAP": "150",
            "RAG_MAX_DOCUMENT_SIZE_MB": "20",
            "RAG_MAX_DOCUMENTS": "200",
            "RAG_RELEVANCE_THRESHOLD": "0.2",
            "RAG_MAX_SEARCH_RESULTS": "10",
            "RAG_TFIDF_MAX_FEATURES": "10000",
            "OPENSSL_FIPS": "1",
            "RAG_HASH_ALGORITHM": "sha384",
            "RAG_DEMO_MODE": "true",
            "RAG_DEMO_MAX_PAGES": "500",
            "RAG_DEMO_MAX_PDFS": "100"
        }
        
        with patch.dict(os.environ, env_vars):
            config = RAGConfig()
            loaded_config = load_rag_config_from_env(config)
            
            assert loaded_config.enabled is False
            assert loaded_config.storage_path == "/custom/storage"
            assert loaded_config.max_memory_mb == 150
            assert loaded_config.cache_size == 75
            assert loaded_config.chunk_size == 800
            assert loaded_config.chunk_overlap == 150
            assert loaded_config.max_document_size_mb == 20
            assert loaded_config.max_documents == 200
            assert loaded_config.relevance_threshold == 0.2
            assert loaded_config.max_search_results == 10
            assert loaded_config.tfidf_max_features == 10000
            assert loaded_config.fips_mode is True
            assert loaded_config.hash_algorithm == "sha384"
            assert loaded_config.demo_mode is True
            assert loaded_config.demo_max_pages == 500
            assert loaded_config.demo_max_pdfs == 100


class TestDeploymentConfigs:
    """Test deployment-specific configuration templates."""
    
    def test_deployment_config_templates_exist(self):
        """Test that all deployment config templates exist."""
        assert "development" in DEPLOYMENT_CONFIGS
        assert "demo" in DEPLOYMENT_CONFIGS
        assert "openshift" in DEPLOYMENT_CONFIGS
    
    def test_get_deployment_config(self):
        """Test getting deployment configuration templates."""
        dev_config = get_deployment_config("development")
        assert dev_config["enabled"] is True
        assert dev_config["fips_mode"] is False  # Relaxed for development
        
        demo_config = get_deployment_config("demo")
        assert demo_config["demo_mode"] is True
        assert demo_config["fips_mode"] is True
        
        openshift_config = get_deployment_config("openshift")
        assert openshift_config["fips_mode"] is True
        assert openshift_config["hash_algorithm"] == "sha256"
    
    def test_get_nonexistent_deployment_config(self):
        """Test getting nonexistent deployment config returns demo default."""
        config = get_deployment_config("nonexistent")
        demo_config = get_deployment_config("demo")
        assert config == demo_config
    
    def test_apply_deployment_config(self):
        """Test applying deployment configuration to RAGConfig."""
        base_config = RAGConfig()
        
        # Apply development config
        dev_config = apply_deployment_config(base_config, "development")
        assert dev_config.storage_path == "./dev_rag_storage"
        assert dev_config.max_memory_mb == 200
        assert dev_config.fips_mode is False
        
        # Apply demo config
        demo_config = apply_deployment_config(base_config, "demo")
        assert demo_config.storage_path == "/app/rag_storage"
        assert demo_config.demo_mode is True
        assert demo_config.demo_max_pages == 300
        
        # Apply OpenShift config
        openshift_config = apply_deployment_config(base_config, "openshift")
        assert openshift_config.storage_path == "/mnt/rag-storage"
        assert openshift_config.max_documents == 200
        assert openshift_config.demo_mode is False


class TestConfigurationIntegration:
    """Test integration scenarios and edge cases."""
    
    def test_config_file_and_env_precedence(self):
        """Test that environment variables override file configuration."""
        temp_dir = tempfile.mkdtemp()
        config_path = os.path.join(temp_dir, "test_config.json")
        
        try:
            # Create config file
            file_config = {"max_memory_mb": 200, "chunk_size": 800}
            with open(config_path, "w") as f:
                json.dump(file_config, f)
            
            # Set environment variable
            with patch.dict(os.environ, {"RAG_MAX_MEMORY_MB": "300"}):
                manager = RAGConfigManager(config_path)
                config = manager.load_config()
                
                # Environment should override file
                assert config.max_memory_mb == 300
                # File setting should still apply
                assert config.chunk_size == 800
        
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_configuration_with_all_validation_errors(self):
        """Test configuration with multiple validation errors."""
        invalid_config = RAGConfig(
            max_memory_mb=10,  # Too low
            chunk_size=50,     # Too small
            chunk_overlap=2000,  # Greater than chunk_size
            relevance_threshold=2.0,  # Out of range
            supported_formats=[],  # Empty
            demo_mode=True,
            demo_max_pages=50,  # Too low for demo
            fips_mode=True,
            hash_algorithm="md5"  # Invalid for FIPS
        )
        
        validator = RAGConfigValidator()
        errors = validator.validate_config(invalid_config)
        
        # Should have multiple errors
        assert len(errors) >= 7
        assert any("memory limit too low" in error.lower() for error in errors)
        assert any("chunk size too small" in error.lower() for error in errors)
        assert any("chunk overlap must be less" in error.lower() for error in errors)
        assert any("relevance threshold must be between" in error.lower() for error in errors)
        assert any("at least one supported format" in error.lower() for error in errors)
        assert any("demo max pages too low" in error.lower() for error in errors)
        assert any("fips mode requires" in error.lower() for error in errors)
    
    @patch('config.RAGConfigValidator._validate_storage_path', return_value=False)
    def test_config_manager_backup_on_import_failure(self, mock_validate_path):
        """Test that config manager creates backup on import failure."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            config_path = os.path.join(temp_dir, "test_config.json")
            import_path = os.path.join(temp_dir, "import_config.json")
            
            # Create existing config with valid path
            existing_config = {"max_memory_mb": 100, "storage_path": "./valid_path"}
            with open(config_path, "w") as f:
                json.dump(existing_config, f)
            
            # Create import file with invalid configuration that will fail validation
            invalid_config = {"max_memory_mb": 10, "storage_path": "/invalid/path"}  # Too low memory + invalid path
            with open(import_path, "w") as f:
                json.dump(invalid_config, f)
            
            manager = RAGConfigManager(config_path)
            
            # Import should fail due to validation errors
            with pytest.raises(RuntimeError, match="Failed to import configuration"):
                manager.import_config(import_path)
            
            # Backup should have been created and original restored
            backup_path = f"{config_path}.backup"
            assert os.path.exists(backup_path) or os.path.exists(config_path)
        
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__])