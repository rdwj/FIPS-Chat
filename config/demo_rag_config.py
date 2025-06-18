#!/usr/bin/env python3
"""
RAG Configuration System Demo

This script demonstrates the comprehensive RAG configuration management system,
including loading, validation, environment variable overrides, and deployment configurations.
"""

import os
import sys
import json
from pathlib import Path

# Add the parent directory to the Python path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    RAGConfig,
    RAGConfigManager,
    RAGConfigValidator,
    get_deployment_config,
    apply_deployment_config,
    DEPLOYMENT_CONFIGS
)


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_config(config: RAGConfig, title: str = "Configuration"):
    """Print configuration in a readable format."""
    print(f"\n{title}:")
    print(f"  Enabled: {config.enabled}")
    print(f"  Storage Path: {config.storage_path}")
    print(f"  Max Memory: {config.max_memory_mb}MB")
    print(f"  Chunk Size: {config.chunk_size}")
    print(f"  Chunk Overlap: {config.chunk_overlap}")
    print(f"  Max Documents: {config.max_documents}")
    print(f"  Max Search Results: {config.max_search_results}")
    print(f"  Relevance Threshold: {config.relevance_threshold}")
    print(f"  FIPS Mode: {config.fips_mode}")
    print(f"  Hash Algorithm: {config.hash_algorithm}")
    print(f"  Demo Mode: {config.demo_mode}")
    if config.demo_mode:
        print(f"  Demo Max Pages: {config.demo_max_pages}")
        print(f"  Demo Max PDFs: {config.demo_max_pdfs}")


def demo_basic_configuration():
    """Demonstrate basic configuration loading."""
    print_header("1. Basic Configuration")
    
    # Load default configuration
    config = RAGConfig()
    print_config(config, "Default Configuration")
    
    # Validate configuration
    validator = RAGConfigValidator()
    errors = validator.validate_config(config)
    if errors:
        print(f"\n❌ Validation errors: {', '.join(errors)}")
    else:
        print("\n✅ Configuration is valid")


def demo_environment_variables():
    """Demonstrate environment variable overrides."""
    print_header("2. Environment Variable Overrides")
    
    # Set some environment variables
    test_env_vars = {
        "RAG_MAX_MEMORY_MB": "150",
        "RAG_CHUNK_SIZE": "800",
        "RAG_DEMO_MODE": "true",
        "RAG_DEMO_MAX_PAGES": "500"
    }
    
    print("Setting environment variables:")
    for key, value in test_env_vars.items():
        print(f"  {key}={value}")
        os.environ[key] = value
    
    # Load configuration with environment overrides
    manager = RAGConfigManager()
    config = manager.load_config()
    print_config(config, "Configuration with Environment Overrides")
    
    # Clean up environment variables
    for key in test_env_vars:
        del os.environ[key]


def demo_deployment_configurations():
    """Demonstrate deployment-specific configurations."""
    print_header("3. Deployment-Specific Configurations")
    
    base_config = RAGConfig()
    
    for deployment_type in ["development", "demo", "openshift"]:
        print(f"\n{deployment_type.upper()} Configuration:")
        deployed_config = apply_deployment_config(base_config, deployment_type)
        print_config(deployed_config, f"{deployment_type.title()} Config")


def demo_configuration_validation():
    """Demonstrate configuration validation."""
    print_header("4. Configuration Validation")
    
    validator = RAGConfigValidator()
    
    # Test valid configuration
    valid_config = RAGConfig()
    errors = validator.validate_config(valid_config)
    print(f"Valid config errors: {len(errors)} ({'✅ PASS' if len(errors) == 0 else '❌ FAIL'})")
    
    # Test invalid configuration
    invalid_config = RAGConfig(
        max_memory_mb=10,  # Too low
        chunk_size=50,     # Too small
        chunk_overlap=2000,  # Greater than chunk_size
        relevance_threshold=2.0,  # Out of range
        supported_formats=[],  # Empty
    )
    
    errors = validator.validate_config(invalid_config)
    print(f"\nInvalid config errors: {len(errors)}")
    for i, error in enumerate(errors, 1):
        print(f"  {i}. {error}")


def demo_configuration_manager():
    """Demonstrate configuration manager features."""
    print_header("5. Configuration Manager")
    
    # Create a temporary config file path
    temp_config_path = "demo_rag_config.json"
    
    try:
        manager = RAGConfigManager(temp_config_path)
        
        # Create and save a custom configuration
        custom_config = RAGConfig(
            max_memory_mb=200,
            chunk_size=800,
            demo_mode=True,
            demo_max_pages=500
        )
        
        print("Saving custom configuration...")
        manager.save_config(custom_config)
        print(f"✅ Configuration saved to {temp_config_path}")
        
        # Load the configuration back
        print("\nLoading configuration from file...")
        loaded_config = manager.load_config()
        print_config(loaded_config, "Loaded Configuration")
        
        # Export configuration
        export_path = "exported_config.json"
        manager.export_config(export_path)
        print(f"\n✅ Configuration exported to {export_path}")
        
        # Show file contents
        print(f"\nConfiguration file contents ({temp_config_path}):")
        with open(temp_config_path, 'r') as f:
            config_data = json.load(f)
            print(json.dumps(config_data, indent=2))
    
    finally:
        # Clean up temporary files
        for file_path in [temp_config_path, "exported_config.json"]:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🗑️  Cleaned up {file_path}")


def demo_memory_and_performance():
    """Demonstrate memory and performance considerations."""
    print_header("6. Memory and Performance Considerations")
    
    configs = {
        "Light": RAGConfig(max_memory_mb=50, cache_size=25, max_documents=50),
        "Standard": RAGConfig(max_memory_mb=100, cache_size=50, max_documents=100),
        "Heavy": RAGConfig(max_memory_mb=200, cache_size=100, max_documents=200)
    }
    
    for name, config in configs.items():
        estimated_memory = (
            config.max_memory_mb +  # RAG processing
            (config.cache_size * 0.5) +  # Document cache (rough estimate)
            (config.max_documents * 0.1)  # Index overhead (rough estimate)
        )
        
        print(f"\n{name} Configuration:")
        print(f"  Max Memory: {config.max_memory_mb}MB")
        print(f"  Cache Size: {config.cache_size} documents")
        print(f"  Max Documents: {config.max_documents}")
        print(f"  Estimated Total Memory: ~{estimated_memory:.1f}MB")


def main():
    """Run all configuration demos."""
    print("🚀 RAG Configuration System Demo")
    print("This demo shows the comprehensive RAG configuration management capabilities.")
    
    try:
        demo_basic_configuration()
        demo_environment_variables()
        demo_deployment_configurations()
        demo_configuration_validation()
        demo_configuration_manager()
        demo_memory_and_performance()
        
        print_header("Demo Complete")
        print("✅ All configuration demos completed successfully!")
        print("\n📚 Next Steps:")
        print("   1. Set your environment variables as needed")
        print("   2. Choose appropriate deployment configuration")
        print("   3. Test your configuration with the validation system")
        print("   4. Monitor memory usage in production")
        print("\n📖 See config/rag_environment_vars.md for detailed documentation")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())