# FIPS Chat

A FIPS 140-2 compliant multi-provider AI chat and image analysis platform designed for containerized deployment in OpenShift environments.

## Features

- **💬 Chat Interface**: Real-time conversations with Ollama models
- **🖼️ Image Analysis**: Upload and analyze images with vision models
- **🎛️ Model Management**: Dynamic model selection and information display
- **📊 Session Statistics**: Track usage, response times, and performance
- **💾 Export Functionality**: Export conversations and analyses
- **⚙️ Configurable Settings**: Adjust temperature, tokens, and other parameters
- **🔒 FIPS Compliance**: Built with FIPS 140-2 cryptographic standards

## 🚨 IMPORTANT: Production Deployment Requirements

**This application MUST be deployed as a container for FIPS compliance. Direct Python execution is only supported for development.**

### Production Deployment

- ✅ **Container deployment** (Podman/Docker + OpenShift)
- ✅ **FIPS-enabled environment**
- ✅ **Container registry access**

### Development Only

- ⚠️ **Local Python execution** (development/testing only)

## Prerequisites

**For Production (Container Deployment):**

- Podman or Docker
- OpenShift 4.8+ cluster with FIPS mode enabled
- Access to container registry (Quay.io, Docker Hub, etc.)
- [Ollama](https://ollama.ai/) service deployed in OpenShift

**For Development Only:**

- Python 3.11+
- [Ollama](https://ollama.ai/) installed and running locally
- At least one Ollama model installed (see recommended models below)

## Quick Start

### Production Deployment (Container)

1. **Build the FIPS-compliant container**:

   ```bash
   ./scripts/build-podman.sh
   ```
2. **Tag and push to registry**:

   ```bash
   podman tag ollama-streamlit:latest quay.io/your-username/fips-chat:latest
   podman push quay.io/your-username/fips-chat:latest
   ```
3. **Deploy to OpenShift**:

   ```bash
   cd openshift/
   # Update image reference in deployment.yaml
   oc apply -k .
   ```

### Development Setup (Local Only)

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
2. **Install Ollama models** (recommended):

   ```bash
   # Vision models for image analysis
   ollama pull llava:7b
   ollama pull granite3.2-vision:latest

   # Chat models for conversations
   ollama pull granite3.3:8b
   ollama pull gemma3:latest
   ollama pull phi4-mini:3.8b

   # Code-focused model
   ollama pull qwen2.5-coder:7b
   ```
3. **Run the application** (development only):

   ```bash
   # Start Ollama service
   ollama serve

   # Run Streamlit application
   streamlit run app.py

   # Open browser to http://localhost:8501
   ```

## Container Testing

Test the container locally before deploying:

```bash
# Test with provided script
./scripts/test-podman.sh

# Or manual testing
podman run -p 8080:8080 --rm ollama-streamlit:latest
# Open browser to http://localhost:8080
```

## Model Management

After deployment, you'll need to install Ollama models. The application provides several ways to manage models:

### 🔗 **Admin Route (Easiest)**

Access the admin interface at: `https://ollama-admin-{namespace}.apps.{cluster}/`

```bash
# Deploy a test model
curl -X POST https://ollama-admin-ollama-platform.apps.your-cluster.com/api/pull \
  -d '{"name": "llama3.2:1b"}' -H "Content-Type: application/json"

# Check available models  
curl -s https://ollama-admin-ollama-platform.apps.your-cluster.com/api/tags
```

### 🔧 **CLI Access (Advanced)**

```bash
# Port forward to Ollama
oc port-forward service/ollama-service 11434:11434 &

# Deploy models
curl -X POST http://localhost:11434/api/pull \
  -d '{"name": "granite3.3:8b"}' -H "Content-Type: application/json"
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete model management guide.

## Recommended Models

### Primary Models (Balanced Performance)

- **llava:7b** - Primary vision model for image description (4.7 GB)
- **granite3.2-vision:latest** - Alternative vision model (2.4 GB)
- **granite3.3:8b** - Primary chat model (4.9 GB)
- **gemma3:latest** - Lightweight chat alternative (3.3 GB)
- **phi4-mini:3.8b** - Fast response chat model (2.5 GB)

### Secondary Models (Power Users)

- **qwen2.5-coder:7b** - Code-focused conversations (4.7 GB)
- **mistral-small3.1:24b** - High-quality responses (15 GB)

## Application Structure

```
fips-chat/
├── app.py                    # Main Streamlit application
├── config.py                # Configuration management
├── ollama_client.py          # Ollama API wrapper
├── ui_components/            # Reusable UI components
│   ├── chat_interface.py     # Chat functionality
│   ├── image_interface.py    # Image analysis
│   └── model_selector.py     # Model management
├── utils/                    # Utility functions
│   ├── image_processing.py   # Image utilities
│   └── session_manager.py    # Session management
├── tests/                    # Test suite
├── requirements.txt          # Dependencies
└── README.md                # This file
```

## Configuration

The application can be configured through environment variables:

```bash
export OLLAMA_HOST="http://localhost:11434"  # Ollama server URL
export DEFAULT_CHAT_MODEL="granite3.3:8b"   # Default chat model
export DEFAULT_VISION_MODEL="llava:7b"      # Default vision model
export TEMPERATURE="0.7"                     # Model temperature
export MAX_TOKENS="2048"                     # Max response length
export MAX_FILE_SIZE_MB="10"                 # Max image file size
```

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_config.py -v
python -m pytest tests/test_ollama_client.py -v
python -m pytest tests/test_image_processing.py -v
```

## Performance Targets

- Model list refresh: < 2 seconds
- Chat response initiation: < 3 seconds
- Image analysis initiation: < 5 seconds
- UI responsiveness: No blocking operations

## Troubleshooting

### Common Issues

1. **"Cannot connect to Ollama"**

   - Ensure Ollama is running: `ollama serve`
   - Check if Ollama is accessible: `ollama list`
2. **"No models available"**

   - Install at least one model: `ollama pull llava:7b`
   - Verify models are installed: `ollama list`
3. **"Model does not support image analysis"**

   - Select a vision model (llava:7b, granite3.2-vision:latest)
   - Check model capabilities in the Models tab
4. **High memory usage warning**

   - Use the "Clean Up" button in the sidebar
   - Clear conversation history
   - Reduce image file sizes

### Performance Tips

- Use smaller models (phi4-mini:3.8b, gemma3:latest) for faster responses
- Reduce image file sizes before upload
- Clear old conversations and images regularly
- Adjust temperature and max tokens for your use case

## OpenShift Deployment

For FIPS-compliant deployment to OpenShift:

```bash
# Build with Podman
./scripts/build-podman.sh

# Test locally
./scripts/test-podman.sh

# Deploy to OpenShift
oc apply -k openshift/
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for comprehensive OpenShift deployment instructions.

### FIPS Compliance

✅ **This application is FIPS 140-2 compliant**

- No weak cryptographic functions (MD5, SHA1, etc.)
- Container runs with OPENSSL_FIPS=1
- Uses OCI-compliant Containerfile for Podman

## License

This project is provided as-is for educational and development purposes.
