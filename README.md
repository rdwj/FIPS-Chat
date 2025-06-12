# FIPS Chat

A multi-provider AI chat and image analysis platform designed for local and OpenShift deployment with FIPS compliance.

## Features

- **💬 Chat Interface**: Real-time conversations with Ollama models
- **🖼️ Image Analysis**: Upload and analyze images with vision models
- **🎛️ Model Management**: Dynamic model selection and information display
- **📊 Session Statistics**: Track usage, response times, and performance
- **💾 Export Functionality**: Export conversations and analyses
- **⚙️ Configurable Settings**: Adjust temperature, tokens, and other parameters

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) installed and running locally
- At least one Ollama model installed (see recommended models below)

## Installation

1. **Clone/Download the project**:
   ```bash
   cd /path/to/FIPS-Chat
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Ollama models** (recommended):
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

## Usage

1. **Start Ollama** (if not already running):
   ```bash
   ollama serve
   ```

2. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

3. **Open your browser** to `http://localhost:8501`

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
├── CLAUDE.md                # Development guidance
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

## Development

See `CLAUDE.md` for development guidance and contribution instructions.

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