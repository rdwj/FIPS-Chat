"""Health check endpoints for the Ollama Streamlit application."""

import streamlit as st
import sys
import os
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ollama_client import get_ollama_client
from config import get_config


def check_health():
    """Comprehensive health check for the application."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "checks": {}
    }
    
    try:
        # Check 1: Application startup
        health_status["checks"]["app_startup"] = {
            "status": "pass",
            "message": "Application started successfully"
        }
        
        # Check 2: Configuration loading
        try:
            config = get_config()
            health_status["checks"]["configuration"] = {
                "status": "pass",
                "message": "Configuration loaded successfully",
                "details": {
                    "ollama_host": config.ollama_host,
                    "timeout": config.request_timeout
                }
            }
        except Exception as e:
            health_status["checks"]["configuration"] = {
                "status": "fail",
                "message": f"Configuration error: {str(e)}"
            }
            health_status["status"] = "unhealthy"
        
        # Check 3: Ollama connectivity (non-blocking)
        try:
            client = get_ollama_client()
            models = client.get_available_models()
            
            if models:
                health_status["checks"]["ollama_connection"] = {
                    "status": "pass",
                    "message": f"Connected to Ollama with {len(models)} models available",
                    "details": {
                        "model_count": len(models),
                        "host": config.ollama_host
                    }
                }
            else:
                health_status["checks"]["ollama_connection"] = {
                    "status": "warn",
                    "message": "Connected to Ollama but no models available",
                    "details": {
                        "model_count": 0,
                        "host": config.ollama_host
                    }
                }
        except Exception as e:
            health_status["checks"]["ollama_connection"] = {
                "status": "warn",
                "message": f"Ollama connection issue: {str(e)}",
                "details": {
                    "host": config.ollama_host,
                    "note": "Application can start without Ollama but functionality will be limited"
                }
            }
            # Don't mark as unhealthy - app can start without Ollama
        
        # Check 4: Memory and resources
        try:
            import psutil
            memory_info = psutil.virtual_memory()
            health_status["checks"]["resources"] = {
                "status": "pass" if memory_info.percent < 90 else "warn",
                "message": f"Memory usage: {memory_info.percent:.1f}%",
                "details": {
                    "memory_percent": memory_info.percent,
                    "available_gb": memory_info.available / (1024**3)
                }
            }
        except ImportError:
            health_status["checks"]["resources"] = {
                "status": "pass",
                "message": "Resource monitoring not available (psutil not installed)"
            }
        except Exception as e:
            health_status["checks"]["resources"] = {
                "status": "warn",
                "message": f"Resource check error: {str(e)}"
            }
        
        # Check 5: File system access
        try:
            test_file = "/tmp/health_check_test"
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            
            health_status["checks"]["filesystem"] = {
                "status": "pass",
                "message": "File system access working"
            }
        except Exception as e:
            health_status["checks"]["filesystem"] = {
                "status": "fail",
                "message": f"File system access error: {str(e)}"
            }
            health_status["status"] = "unhealthy"
        
        return health_status
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": f"Health check failed: {str(e)}",
            "checks": health_status.get("checks", {})
        }


def check_readiness():
    """Readiness check - determines if the application is ready to serve traffic."""
    readiness_status = {
        "status": "ready",
        "timestamp": datetime.now().isoformat(),
        "checks": {}
    }
    
    try:
        # Check 1: Configuration is valid
        try:
            config = get_config()
            readiness_status["checks"]["configuration"] = {
                "status": "pass",
                "message": "Configuration loaded"
            }
        except Exception as e:
            readiness_status["checks"]["configuration"] = {
                "status": "fail",
                "message": f"Configuration error: {str(e)}"
            }
            readiness_status["status"] = "not_ready"
        
        # Check 2: Can create Ollama client
        try:
            client = get_ollama_client()
            readiness_status["checks"]["ollama_client"] = {
                "status": "pass",
                "message": "Ollama client initialized"
            }
        except Exception as e:
            readiness_status["checks"]["ollama_client"] = {
                "status": "fail",
                "message": f"Ollama client error: {str(e)}"
            }
            readiness_status["status"] = "not_ready"
        
        # Check 3: Streamlit session state can be initialized
        try:
            # Simple session state test
            if "health_check" not in st.session_state:
                st.session_state.health_check = True
            
            readiness_status["checks"]["session_state"] = {
                "status": "pass",
                "message": "Session state accessible"
            }
        except Exception as e:
            readiness_status["checks"]["session_state"] = {
                "status": "fail",
                "message": f"Session state error: {str(e)}"
            }
            readiness_status["status"] = "not_ready"
        
        return readiness_status
        
    except Exception as e:
        return {
            "status": "not_ready",
            "timestamp": datetime.now().isoformat(),
            "error": f"Readiness check failed: {str(e)}",
            "checks": readiness_status.get("checks", {})
        }


def render_health_page():
    """Render a health status page for debugging."""
    st.set_page_config(
        page_title="Health Check",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Application Health Status")
    
    # Health check
    st.subheader("Health Check")
    health = check_health()
    
    if health["status"] == "healthy":
        st.success("✅ Application is healthy")
    elif health["status"] == "unhealthy":
        st.error("❌ Application is unhealthy")
    else:
        st.warning("⚠️ Application has warnings")
    
    # Readiness check
    st.subheader("Readiness Check")
    readiness = check_readiness()
    
    if readiness["status"] == "ready":
        st.success("✅ Application is ready")
    else:
        st.error("❌ Application is not ready")
    
    # Detailed status
    st.subheader("Detailed Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Health Checks:**")
        for check_name, check_result in health.get("checks", {}).items():
            status_icon = {
                "pass": "✅",
                "warn": "⚠️",
                "fail": "❌"
            }.get(check_result["status"], "❓")
            
            st.write(f"{status_icon} {check_name}: {check_result['message']}")
    
    with col2:
        st.write("**Readiness Checks:**")
        for check_name, check_result in readiness.get("checks", {}).items():
            status_icon = {
                "pass": "✅",
                "warn": "⚠️", 
                "fail": "❌"
            }.get(check_result["status"], "❓")
            
            st.write(f"{status_icon} {check_name}: {check_result['message']}")
    
    # Raw JSON for debugging
    with st.expander("Raw Health Data"):
        st.json(health)
    
    with st.expander("Raw Readiness Data"):
        st.json(readiness)


if __name__ == "__main__":
    render_health_page()