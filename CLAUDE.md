# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Streamlit application that integrates with Ollama models to provide dual functionality: text-based chat conversations and image description capabilities. The application is designed for local deployment on macOS with optimization for mid-range hardware.

## Development Commands

### Running the Application
```bash
streamlit run app.py
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Testing
```bash
python -m pytest tests/
python -m pytest tests/unit/     # Unit tests only
python -m pytest tests/integration/  # Integration tests only
```

## Architecture Overview

### Core Components
- **app.py**: Main Streamlit application entry point
- **ollama_client.py**: Ollama API integration wrapper
- **config.py**: Configuration management for models and settings
- **ui_components/**: Modular UI components for chat, image analysis, and model selection
- **utils/**: Utility functions for image processing and session management

### Key Integrations
- **Ollama API**: Uses `/api/generate`, `/api/chat`, `/api/tags`, and `/api/show` endpoints
- **Image Processing**: Supports JPEG, PNG, WebP, GIF with Pillow
- **Session State**: Manages conversation history and uploaded images within Streamlit sessions

### Recommended Models
- **Primary Chat**: granite3.3:8b (4.9 GB), gemma3:latest (3.3 GB), phi4-mini:3.8b (2.5 GB)
- **Vision Models**: llava:7b (4.7 GB), granite3.2-vision:latest (2.4 GB)
- **Code-focused**: qwen2.5-coder:7b (4.7 GB)

## Performance Requirements

- Chat response initiation: < 3 seconds
- Image analysis initiation: < 5 seconds
- Model list refresh: < 2 seconds
- Request timeout: 30-60 seconds with retry logic

## Error Handling Patterns

The application must gracefully handle:
- Ollama service unavailable
- Model loading failures
- Image upload issues
- Network connectivity problems
- Insufficient system resources

Always provide clear error messages with suggested actions and maintain UI responsiveness during async operations.