# Ollama Streamlit Application Specification

## Project Overview

Develop a local Streamlit application for macOS that integrates with Ollama models to provide dual functionality: text-based chat conversations and image description capabilities. The application should be optimized for mid-range hardware specifications while maintaining high performance and user experience.

## Core Requirements

### 1. Application Architecture
- **Framework**: Streamlit for the web interface
- **Backend**: Ollama API integration via HTTP requests
- **Local Deployment**: Self-contained application running on localhost
- **Platform**: macOS optimized (Intel and Apple Silicon compatible)

### 2. Functional Requirements

#### 2.1 Chat Interface
- Real-time conversational interface with selected Ollama models
- Message history persistence within session
- Support for multi-turn conversations
- Message formatting with proper typography and code highlighting
- Copy-to-clipboard functionality for responses
- Clear conversation history option

#### 2.2 Image Analysis Interface
- Image upload functionality (drag-and-drop and file picker)
- Support for common image formats: JPEG, PNG, WebP, GIF
- Real-time image preview with metadata display
- Vision model integration for image description
- Ability to ask follow-up questions about uploaded images
- Image history within session

#### 2.3 Model Management
- Dynamic model selection dropdown
- Model information display (size, capabilities)
- Automatic detection of available Ollama models
- Performance indicators (response time, model load status)
- Model switching without application restart

## Technical Specifications

### 3. Recommended Model Selection

Based on the provided model list and hardware optimization, prioritize these models:

#### Primary Models (Recommended for balanced performance):
- **llava:7b** - Primary vision model for image description (4.7 GB)
- **granite3.2-vision:latest** - Alternative vision model (2.4 GB) 
- **granite3.3:8b** - Primary chat model (4.9 GB)
- **gemma3:latest** - Lightweight chat alternative (3.3 GB)
- **phi4-mini:3.8b** - Fast response chat model (2.5 GB)

#### Secondary Models (Power user options):
- **qwen2.5-coder:7b** - Code-focused conversations (4.7 GB)
- **mistral-small3.1:24b** - High-quality responses when resources allow (15 GB)

### 4. User Interface Design

#### 4.1 Layout Structure
```
Header: App Title + Model Selector + Settings
Main Content: 
  - Tab 1: Chat Interface
  - Tab 2: Image Analysis
  - Tab 3: Model Information
Sidebar: 
  - Active model status
  - Performance metrics
  - Session controls
```

#### 4.2 Chat Interface Components
- Message input field with send button
- Chat history display with user/assistant message distinction
- Typing indicator during model response generation
- Response streaming (if supported by Ollama API)
- Message timestamp display
- Export conversation option

#### 4.3 Image Analysis Components
- Image upload area with visual feedback
- Image display panel with zoom/pan capabilities
- Analysis results panel
- Follow-up question input
- Image metadata display (dimensions, file size, format)

### 5. API Integration Requirements

#### 5.1 Ollama API Endpoints
- `/api/generate` - For text generation (chat)
- `/api/chat` - For conversational context
- `/api/tags` - For model listing
- `/api/show` - For model information

#### 5.2 Request Handling
- Asynchronous request processing
- Error handling with user-friendly messages
- Request timeout configuration (30-60 seconds)
- Retry logic for failed requests
- Progress indicators for long-running requests

### 6. Performance Requirements

#### 6.1 Response Time Targets
- Model list refresh: < 2 seconds
- Chat response initiation: < 3 seconds
- Image analysis initiation: < 5 seconds
- UI responsiveness: No blocking operations

#### 6.2 Resource Management
- Memory usage monitoring
- Model loading status indicators
- Automatic cleanup of temporary files
- Session state management

### 7. Error Handling & User Experience

#### 7.1 Error Scenarios
- Ollama service unavailable
- Model not found or failed to load
- Image upload failures
- Network connectivity issues
- Insufficient system resources

#### 7.2 User Feedback
- Clear error messages with suggested actions
- Loading states for all async operations
- Success confirmations for user actions
- Help tooltips for complex features

### 8. Configuration & Settings

#### 8.1 Application Settings
- Default model selection
- Response generation parameters (temperature, max tokens)
- UI theme options (light/dark mode)
- Image processing settings (max file size, supported formats)

#### 8.2 Model Parameters
- Temperature control (0.1 - 1.0)
- Max tokens/response length
- System prompt customization
- Context window management

### 9. Development Requirements

#### 9.1 Dependencies
```python
streamlit >= 1.28.0
requests >= 2.31.0
pillow >= 10.0.0
python-multipart >= 0.0.6
ollama >= 0.1.0  # Official Ollama Python client
```

#### 9.2 File Structure
```
ollama-streamlit-app/
├── app.py                 # Main Streamlit application
├── config.py             # Configuration management
├── ollama_client.py      # Ollama API wrapper
├── ui_components/        # Reusable UI components
│   ├── chat_interface.py
│   ├── image_interface.py
│   └── model_selector.py
├── utils/               # Utility functions
│   ├── image_processing.py
│   └── session_manager.py
├── requirements.txt     # Python dependencies
└── README.md           # Setup and usage instructions
```

### 10. Testing Requirements

#### 10.1 Unit Tests
- Ollama API client functionality
- Image processing utilities
- Configuration management
- Error handling scenarios

#### 10.2 Integration Tests
- End-to-end chat workflows
- Image upload and analysis workflows
- Model switching functionality
- Session persistence

### 11. Documentation Requirements

#### 11.1 User Documentation
- Installation and setup guide
- Feature overview with screenshots
- Troubleshooting guide
- Model selection recommendations

#### 11.2 Developer Documentation
- Code architecture overview
- API integration patterns
- Extension guidelines
- Deployment instructions

### 12. Future Enhancement Considerations

#### 12.1 Planned Features
- Conversation export (PDF, markdown)
- Custom model fine-tuning integration
- Multi-image analysis capabilities
- Plugin system for additional models
- Performance analytics dashboard

#### 12.2 Scalability Considerations
- Docker containerization support
- Multi-user session management
- Database integration for conversation persistence
- API rate limiting and queuing

## Success Criteria

1. **Functionality**: Both chat and image analysis features work reliably with recommended models
2. **Performance**: Responsive UI with sub-5-second response times for typical queries
3. **Usability**: Intuitive interface requiring no technical knowledge to operate
4. **Reliability**: Graceful handling of common error scenarios
5. **Compatibility**: Runs consistently on both Intel and Apple Silicon Macs

## Acceptance Testing

- [ ] Chat interface handles multi-turn conversations correctly
- [ ] Image upload and analysis works with all supported formats
- [ ] Model switching functions without data loss
- [ ] Application recovers gracefully from Ollama service interruptions
- [ ] All UI components are responsive and accessible
- [ ] Memory usage remains stable during extended sessions
- [ ] Error messages are clear and actionable