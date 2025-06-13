# FIPS Chat Refactoring: API-First Architecture

## Project Goal
Refactor FIPS Chat from Ollama-centric design to a generic API-first architecture that discovers models dynamically from any compatible endpoint.

## Current State Analysis
- Ollama deeply integrated in `ollama_client.py`, `app.py`, and `config.py`
- Conditional UI logic based on provider selection
- Hardcoded model recommendations
- Separate code paths for Ollama vs external APIs

## Target State
- Single unified API client
- Dynamic model discovery
- Provider-agnostic configuration
- Simplified UI without conditional logic
- Optimized for OpenShift deployment

---

## Implementation Plan

### Phase 1: Backend Unification
**Goal**: Create unified API client with dynamic model discovery

#### 1.1 Create UnifiedAPIClient Class
- [ ] Design new `ai_client.py` with provider-agnostic interface
- [ ] Implement base client with common methods: `chat()`, `generate_with_image()`, `discover_models()`
- [ ] Add provider detection logic for OpenAI-compatible, Anthropic, custom APIs
- [ ] Create adapter pattern for different API response formats

#### 1.2 Implement Model Discovery
- [ ] Add `/models` endpoint detection and parsing
- [ ] Implement model capability detection (chat, vision, code)
- [ ] Create fallback mechanisms for APIs without model listing
- [ ] Add model caching with configurable refresh intervals
- [ ] Handle various model list response formats

#### 1.3 Provider Detection & Configuration
- [ ] Auto-detect API type from endpoint responses
- [ ] Create provider-specific configuration templates
- [ ] Implement authentication handling for different providers
- [ ] Add connection testing with detailed error reporting

#### 1.4 Testing & Validation
- [ ] Test with vLLM endpoints
- [ ] Test with OpenAI-compatible APIs
- [ ] Test with custom/agentic pipeline endpoints
- [ ] Verify backward compatibility with existing session data

### Phase 2: Configuration Migration
**Goal**: Remove Ollama-specific configuration and make system provider-agnostic

#### 2.1 Update config.py
- [ ] Remove `ollama_host` and Ollama-specific settings
- [ ] Remove hardcoded `RECOMMENDED_MODELS` list
- [ ] Keep only generic settings (timeouts, file sizes, UI preferences)
- [ ] Add dynamic model recommendation based on discovery

#### 2.2 Environment Variables
- [ ] Remove `OLLAMA_HOST`, `DEFAULT_CHAT_MODEL`, `DEFAULT_VISION_MODEL`
- [ ] Add generic `API_ENDPOINT`, `API_KEY`, `API_TYPE` defaults
- [ ] Update container environment variable documentation

#### 2.3 Session State Cleanup
- [ ] Unify model selection session keys
- [ ] Remove provider-specific session state variables
- [ ] Standardize conversation history format
- [ ] Add migration logic for existing sessions

### Phase 3: UI Simplification
**Goal**: Remove conditional UI logic and streamline user experience

#### 3.1 Refactor app.py Main Interface
- [ ] Remove provider-specific conditional rendering
- [ ] Simplify sidebar to focus on endpoint configuration
- [ ] Make model selection dynamic based on discovered models
- [ ] Streamline connection status display

#### 3.2 Update UI Components
- [ ] Refactor `model_selector.py` to work with discovered models
- [ ] Update `chat_interface.py` to use unified client
- [ ] Update `image_interface.py` to use unified client
- [ ] Remove Ollama-specific UI elements

#### 3.3 Improve User Experience
- [ ] Add model discovery progress indicators
- [ ] Improve error messages for connection failures
- [ ] Add model capability indicators (chat/vision/code)
- [ ] Implement model search/filtering

### Phase 4: OpenShift Optimization
**Goal**: Optimize deployment for API-only architecture

#### 4.1 Update Deployment Configurations
- [ ] Remove Ollama service dependencies from `openshift/` configs
- [ ] Update resource requirements for lighter API-only client
- [ ] Simplify network policies (no internal Ollama communication needed)
- [ ] Update ConfigMap templates

#### 4.2 Container Optimization
- [ ] Review and potentially reduce container resource requirements
- [ ] Update health checks if needed
- [ ] Verify FIPS compliance with new architecture
- [ ] Update build scripts documentation

#### 4.3 Documentation Updates
- [ ] Update CLAUDE.md with new architecture
- [ ] Update README.md deployment instructions
- [ ] Create API endpoint configuration examples
- [ ] Document model discovery process

---

## Testing Strategy

### Unit Tests
- [ ] Test UnifiedAPIClient with mock endpoints
- [ ] Test model discovery with various API response formats
- [ ] Test provider detection logic
- [ ] Test session state migration

### Integration Tests
- [ ] Test full workflow with real API endpoints
- [ ] Test model discovery across different providers
- [ ] Test chat and image analysis functionality
- [ ] Test error handling and recovery

### Deployment Tests
- [ ] Test container build and deployment
- [ ] Test OpenShift deployment with new configuration
- [ ] Verify FIPS compliance in deployed environment
- [ ] Load test with multiple concurrent users

---

## Migration Considerations

### Backward Compatibility
- Existing user sessions should continue working
- Configuration migration for existing deployments
- Graceful handling of deprecated environment variables

### Rollback Plan
- Keep original files with `.bak` extension during development
- Maintain feature flag for switching between old/new implementations
- Document rollback procedure

### Performance Impact
- Model discovery may add initial latency
- Implement caching to minimize repeated API calls
- Monitor response times compared to current implementation

---

## Success Criteria

1. **Functionality**: All current features work with new architecture
2. **Performance**: Response times within 10% of current implementation
3. **Reliability**: Error handling improved with better user feedback
4. **Maintainability**: Reduced code complexity and conditional logic
5. **Deployment**: Successful OpenShift deployment without Ollama dependencies
6. **FIPS Compliance**: Maintained throughout refactoring

---

## Timeline Estimate

- **Phase 1**: 2-3 days (Backend unification)
- **Phase 2**: 1 day (Configuration migration)  
- **Phase 3**: 2 days (UI simplification)
- **Phase 4**: 1 day (OpenShift optimization)
- **Testing & Polish**: 1-2 days

**Total**: 7-9 days

---

## Risk Mitigation

1. **API Compatibility**: Test with multiple endpoint types early
2. **Model Discovery Failures**: Implement robust fallback mechanisms
3. **Performance Regression**: Benchmark before/after implementation
4. **User Experience**: Gather feedback on UI changes
5. **Deployment Issues**: Test in staging environment before production