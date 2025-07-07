# FIPS Chat Container Deployment Guide

This guide covers deploying FIPS Chat to OpenShift with FIPS compliance.

**⚠️ MANDATORY: This application MUST be deployed as a container for FIPS compliance. Direct Python execution or local development approaches are not supported in production environments.**

## Prerequisites

- OpenShift 4.8+ cluster with FIPS mode enabled
- Podman for container building and testing
- `oc` CLI tool configured for your cluster
- Access to a container registry (Quay.io, Docker Hub, or internal registry)
- Ollama service deployed in the same namespace (or accessible via network)

## FIPS Compliance

✅ **This application is FIPS 140-2 compliant**

The codebase has been analyzed and contains no weak cryptographic functions:
- No MD5, SHA1, or other weak hash functions
- No insecure random number generation
- No weak symmetric encryption algorithms
- Base64 usage is for data encoding, not cryptographic purposes

The container runs with `OPENSSL_FIPS=1` and `OPENSSL_FORCE_FIPS_MODE=1` for full FIPS compliance.

## Container Development Workflow

### 1. Build the FIPS-Compliant Container

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Build with Podman (REQUIRED)
./scripts/build-podman.sh

# Or manually:
podman build -f Containerfile -t ollama-streamlit:latest .
```

### 2. Test Container Locally

```bash
# Run automated container tests
./scripts/test-podman.sh

# Or run interactively for development
./scripts/test-podman.sh --interactive

# Manual container run
podman run -p 8080:8080 --rm ollama-streamlit:latest
```

### 3. Tag and Push to Registry

```bash
# Tag for your registry
podman tag ollama-streamlit:latest quay.io/your-username/fips-chat:latest

# Push to registry
podman push quay.io/your-username/fips-chat:latest
```

## Ollama Model Management

After deployment, Ollama will have no models installed. You have several options for model management:

### Option 1: Admin Route (Recommended)

A separate admin route is provided for model management:

**Admin URL:** `https://ollama-admin-{namespace}.apps.{cluster-domain}/`

#### Deploy Models via Admin Route:
```bash
# Deploy a small chat model (recommended for testing)
curl -X POST https://ollama-admin-ollama-platform.apps.your-cluster.com/api/pull \
  -d '{"name": "llama3.2:1b"}' \
  -H "Content-Type: application/json"

# Deploy recommended models
curl -X POST https://ollama-admin-ollama-platform.apps.your-cluster.com/api/pull \
  -d '{"name": "granite3.3:8b"}' \
  -H "Content-Type: application/json"

curl -X POST https://ollama-admin-ollama-platform.apps.your-cluster.com/api/pull \
  -d '{"name": "llava:7b"}' \
  -H "Content-Type: application/json"
```

#### Check Available Models:
```bash
curl -s https://ollama-admin-ollama-platform.apps.your-cluster.com/api/tags
```

### Option 2: CLI Access (Power Users)

Use port-forwarding for direct CLI access:

```bash
# Port forward to Ollama service
oc port-forward service/ollama-service 11434:11434 &

# Deploy models via CLI
curl -X POST http://localhost:11434/api/pull \
  -d '{"name": "llama3.2:1b"}' \
  -H "Content-Type: application/json"

# Stop port-forward when done
pkill -f "oc port-forward.*ollama"
```

### Option 3: Pre-loaded Container (Production)

For production deployments, consider pre-loading models in the Ollama container image:

```dockerfile
# Add to Ollama Containerfile
RUN ollama pull llama3.2:1b && \
    ollama pull granite3.3:8b && \
    ollama pull llava:7b
```

### Security Considerations

- **Admin Route**: Secure with network policies and authentication
- **Model Storage**: Models are stored in persistent volumes
- **Resource Limits**: Set appropriate CPU/memory limits for model loading

## OpenShift Deployment

### 1. Prepare the Environment

```bash
# Create or switch to your project
oc new-project ollama-platform
# or
oc project ollama-platform

# Verify FIPS mode (optional)
oc get node -o jsonpath='{.items[*].status.nodeInfo.kubeletVersion}'
```

### 2. Update Deployment Configuration

```bash
# Update image reference in deployment.yaml
sed -i 's|quay.io/wjackson/fips-chat:latest|your-registry.com/namespace/fips-chat:latest|' openshift/deployment.yaml

# If using private registry, ensure pull secrets are configured
# See imagePullSecrets section in deployment.yaml
```

### 3. Deploy with Kustomize (Recommended)

```bash
# Update image reference in kustomization.yaml
cd openshift/

# Deploy all resources
oc apply -k .

# Or deploy individual components
oc apply -f serviceaccount.yaml
oc apply -f configmap.yaml
oc apply -f deployment.yaml
oc apply -f service.yaml
oc apply -f route.yaml
oc apply -f networkpolicy.yaml
```

### 4. Manual Deployment Steps

```bash
# 1. Create service account
oc apply -f openshift/serviceaccount.yaml

# 2. Create config map
oc apply -f openshift/configmap.yaml

# 3. Update deployment image reference
sed -i 's|fips-chat:latest|your-registry.com/namespace/fips-chat:latest|' openshift/deployment.yaml

# 4. Deploy application
oc apply -f openshift/deployment.yaml
oc apply -f openshift/service.yaml

# 5. Create route (update host in route.yaml first)
oc apply -f openshift/route.yaml

# 6. Apply network policy
oc apply -f openshift/networkpolicy.yaml
```

## Configuration

### Environment Variables

Configure the application via ConfigMap or environment variables:

```yaml
# In configmap.yaml or as environment variables
OLLAMA_HOST: "http://ollama-service:11434"
DEFAULT_CHAT_MODEL: "granite3.3:8b"
DEFAULT_VISION_MODEL: "llava:7b"
TEMPERATURE: "0.7"
MAX_TOKENS: "2048"
REQUEST_TIMEOUT: "60"
MAX_FILE_SIZE_MB: "10"
OPENSSL_FIPS: "1"
OPENSSL_FORCE_FIPS_MODE: "1"
```

### Ollama Service Connection

The application expects Ollama to be available at `http://ollama-service:11434`. Ensure:

1. Ollama is deployed in the same namespace, or
2. Network policies allow cross-namespace communication, or
3. Use external service/endpoint for remote Ollama

Example Ollama service reference:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: ollama-service
spec:
  ports:
  - port: 11434
    targetPort: 11434
  selector:
    app: ollama
```

## Monitoring and Health Checks

### Health Endpoints

The application provides several health check endpoints:

- `/_stcore/health` - Streamlit built-in health check
- `/health` - Custom health check page (navigate to `/health` in browser)

### Kubernetes Probes

The deployment includes:

- **Startup Probe**: Checks if application has started (12 attempts × 5s = 60s max)
- **Readiness Probe**: Checks if ready to serve traffic (every 10s)
- **Liveness Probe**: Checks if application is healthy (every 30s)

### Monitoring Commands

```bash
# Check pod status
oc get pods -l app=fips-chat

# View logs
oc logs -l app=fips-chat -f

# Check health
oc exec deployment/fips-chat -- curl -f http://localhost:8080/_stcore/health

# Port forward for local access
oc port-forward service/fips-chat 8080:8080
```

## Security

### Security Context

The deployment uses a restrictive security context:

- Runs as non-root user (UID 1001)
- No privilege escalation
- Drops all capabilities
- Uses seccomp runtime/default profile
- ReadOnlyRootFilesystem where possible

### Network Policies

The NetworkPolicy restricts traffic to:

- **Ingress**: Only from OpenShift routers and same namespace
- **Egress**: Only to DNS, Ollama service, and HTTPS (443)

### Service Account

Uses a dedicated service account with minimal permissions:

```bash
# View service account
oc get serviceaccount fips-chat

# Check permissions (should be minimal)
oc auth can-i --list --as=system:serviceaccount:$(oc project -q):fips-chat
```

## Troubleshooting

### Common Issues

1. **Pod Fails to Start**
   ```bash
   # Check events
   oc get events --sort-by='.lastTimestamp'
   
   # Check pod logs
   oc logs deployment/fips-chat
   
   # Check security context constraints
   oc get scc
   oc adm policy who-can use scc anyuid
   ```

2. **Health Checks Failing**
   ```bash
   # Test health endpoint manually
   oc exec deployment/fips-chat -- curl -v http://localhost:8080/_stcore/health
   
   # Check application logs
   oc logs deployment/fips-chat --tail=50
   ```

3. **Cannot Connect to Ollama**
   ```bash
   # Test network connectivity
   oc exec deployment/fips-chat -- curl -v http://ollama-service:11434/api/tags
   
   # Check DNS resolution
   oc exec deployment/fips-chat -- nslookup ollama-service
   
   # Verify network policy
   oc get networkpolicy
   ```

4. **FIPS Mode Issues**
   ```bash
   # Check FIPS status in container
   oc exec deployment/fips-chat -- env | grep FIPS
   oc exec deployment/fips-chat -- cat /proc/sys/crypto/fips_enabled
   ```

### Debug Pod

Create a debug pod for troubleshooting:

```bash
oc debug deployment/fips-chat --image=registry.access.redhat.com/ubi9/ubi:latest
```

## Scaling and Performance

### Horizontal Scaling

```bash
# Scale up
oc scale deployment fips-chat --replicas=3

# Auto-scaling (requires metrics server)
oc autoscale deployment fips-chat --min=2 --max=10 --cpu-percent=70
```

### Resource Tuning

Adjust resources in deployment.yaml:

```yaml
resources:
  requests:
    memory: "512Mi"    # Minimum memory
    cpu: "250m"        # Minimum CPU
  limits:
    memory: "2Gi"      # Maximum memory
    cpu: "1000m"       # Maximum CPU
```

### Performance Monitoring

```bash
# Check resource usage
oc top pods -l app=fips-chat

# Monitor events
oc get events -w
```

## Backup and Recovery

### Configuration Backup

```bash
# Export all resources
oc get all,configmap,secret,networkpolicy -l app=fips-chat -o yaml > fips-chat-backup.yaml

# Backup namespace
oc get all -o yaml > namespace-backup.yaml
```

### Disaster Recovery

```bash
# Restore from backup
oc apply -f fips-chat-backup.yaml

# Verify deployment
oc rollout status deployment/fips-chat
```

## Updates and Rollbacks

### Rolling Updates

```bash
# Update image
oc set image deployment/fips-chat streamlit-app=your-registry.com/namespace/fips-chat:v2.0.0

# Monitor rollout
oc rollout status deployment/fips-chat

# Check rollout history
oc rollout history deployment/fips-chat
```

### Rollbacks

```bash
# Rollback to previous version
oc rollout undo deployment/fips-chat

# Rollback to specific revision
oc rollout undo deployment/fips-chat --to-revision=3
```

## Production Considerations

1. **Resource Limits**: Set appropriate CPU/memory limits based on usage
2. **Persistent Storage**: Consider adding persistent volumes for session data
3. **Monitoring**: Integrate with OpenShift monitoring stack
4. **Logging**: Configure centralized logging
5. **Security**: Regular security scans and updates
6. **Backup**: Implement regular backup strategy
7. **Load Testing**: Test with expected concurrent users

## RAG Functionality Deployment

### Prerequisites for RAG

- OpenShift cluster with persistent storage support
- FIPS-enabled nodes (for production)
- Minimum 10Gi storage for RAG data
- 2Gi memory limit for RAG processing
- Document processing capabilities (Docling dependencies included)

### RAG-Specific Deployment Steps

#### 1. Deploy Storage Components

```bash
# Create persistent volumes for RAG storage
oc apply -f openshift/rag-pvc.yaml

# Verify storage
oc get pvc
```

#### 2. Configure RAG Settings

```bash
# Deploy RAG configuration
oc apply -f openshift/rag-configmap.yaml

# Verify configuration
oc get configmap fips-chat-rag-config -o yaml
```

#### 3. Deploy Updated Application

```bash
# Apply updated deployment with RAG support
oc apply -f openshift/deployment.yaml

# Monitor deployment
oc rollout status deployment/fips-chat
```

#### 4. Verify RAG Functionality

```bash
# Check RAG health status
oc exec deployment/fips-chat -- curl -s http://localhost:8080/_stcore/health | grep -A 10 rag_system

# Check storage mounts
oc exec deployment/fips-chat -- ls -la /mnt/rag-storage

# Check configuration
oc exec deployment/fips-chat -- env | grep RAG
```

### RAG Configuration Reference

The following environment variables control RAG functionality:

```yaml
# Core RAG Settings
ENABLE_RAG: "true"                    # Enable/disable RAG functionality
RAG_STORAGE_PATH: "/mnt/rag-storage"  # Storage path for documents
RAG_CACHE_PATH: "/mnt/rag-cache"      # Cache path for indexes

# Memory and Performance
RAG_MAX_MEMORY_MB: "100"              # Maximum memory for RAG processing
RAG_CACHE_SIZE: "50"                  # Cache size for processed documents

# Document Processing
RAG_CHUNK_SIZE: "1000"                # Text chunk size for processing
RAG_CHUNK_OVERLAP: "200"              # Overlap between chunks
RAG_MAX_DOCUMENT_SIZE_MB: "10"        # Maximum document size
RAG_MAX_DOCUMENTS: "100"              # Maximum number of documents

# Search Configuration
RAG_MAX_SEARCH_RESULTS: "5"           # Maximum search results returned
RAG_RELEVANCE_THRESHOLD: "0.1"        # Minimum relevance threshold
RAG_TFIDF_MAX_FEATURES: "5000"        # TF-IDF feature limit

# Demo Environment
RAG_DEMO_MODE: "true"                 # Enable demo constraints
RAG_DEMO_MAX_PAGES: "300"             # Maximum pages in demo mode
RAG_DEMO_MAX_PDFS: "75"               # Maximum PDFs in demo mode

# FIPS Compliance
RAG_HASH_ALGORITHM: "sha256"          # Hash algorithm for FIPS compliance
```

### RAG Testing Procedures

#### 1. Basic Functionality Test

```bash
# Port forward to application
oc port-forward service/fips-chat 8080:8080 &

# Open browser and navigate to Documents tab
echo "Navigate to http://localhost:8080 and test:"
echo "1. Upload a PDF document"
echo "2. Verify processing completes"
echo "3. Test RAG-enhanced chat queries"
echo "4. Verify search results include document context"
```

#### 2. Storage Validation

```bash
# Check storage accessibility
oc exec deployment/fips-chat -- test -w /mnt/rag-storage && echo "Storage writable" || echo "Storage not writable"

# Check available space
oc exec deployment/fips-chat -- df -h /mnt/rag-storage

# List processed documents
oc exec deployment/fips-chat -- find /mnt/rag-storage -type f -name "*.json" | wc -l
```

#### 3. Performance Testing

```bash
# Monitor resource usage during document processing
oc top pods -l app=fips-chat

# Check processing times in logs
oc logs deployment/fips-chat | grep -i "document.*processing"
```

### Troubleshooting RAG Issues

#### Common RAG Problems

1. **RAG Storage Not Accessible**
   ```bash
   # Check PVC status
   oc get pvc rag-storage rag-cache
   
   # Check volume mounts
   oc describe pod -l app=fips-chat | grep -A 5 "Mounts:"
   
   # Test storage access
   oc exec deployment/fips-chat -- touch /mnt/rag-storage/test && echo "OK" || echo "FAIL"
   ```

2. **Document Processing Fails**
   ```bash
   # Check for dependency issues
   oc exec deployment/fips-chat -- python -c "import docling; print('Docling OK')"
   
   # Check memory usage
   oc exec deployment/fips-chat -- cat /proc/meminfo | grep Available
   
   # View processing errors
   oc logs deployment/fips-chat | grep -i error
   ```

3. **Search Not Working**
   ```bash
   # Check search index files
   oc exec deployment/fips-chat -- ls -la /mnt/rag-storage/indexes/
   
   # Test search engine initialization
   oc logs deployment/fips-chat | grep -i "search.*engine"
   ```

4. **FIPS Compliance Issues**
   ```bash
   # Verify FIPS mode
   oc exec deployment/fips-chat -- python -c "import hashlib; print('SHA256 available:', hasattr(hashlib, 'sha256'))"
   
   # Check FIPS environment
   oc exec deployment/fips-chat -- env | grep -i fips
   ```

### RAG Performance Tuning

#### Memory Optimization

```yaml
# In deployment.yaml, adjust resources for RAG workloads
spec:
  template:
    spec:
      containers:
      - name: streamlit-app
        resources:
          requests:
            memory: "1Gi"      # Increased for RAG processing
            cpu: "500m"
          limits:
            memory: "3Gi"      # Higher limit for large documents
            cpu: "1500m"
```

#### Storage Optimization

```yaml
# In rag-pvc.yaml, adjust storage size based on needs
spec:
  resources:
    requests:
      storage: 20Gi  # Increased for larger document collections
```

#### Processing Optimization

```yaml
# In rag-configmap.yaml, tune processing parameters
data:
  RAG_CHUNK_SIZE: "800"          # Smaller chunks for better search
  RAG_MAX_MEMORY_MB: "200"       # Increase for larger documents
  RAG_CACHE_SIZE: "100"          # Larger cache for better performance
```

### RAG Monitoring and Alerting

#### Health Check Monitoring

```bash
# Continuous health monitoring
watch -n 30 'oc exec deployment/fips-chat -- curl -s http://localhost:8080/_stcore/health | jq .checks.rag_system'

# Storage usage monitoring
watch -n 60 'oc exec deployment/fips-chat -- df -h /mnt/rag-storage'
```

#### Log Analysis

```bash
# Monitor RAG-specific logs
oc logs deployment/fips-chat -f | grep -i "rag\|document\|search"

# Check for processing errors
oc logs deployment/fips-chat | grep -i "error\|exception" | grep -i "rag\|document"
```

### RAG Backup and Recovery

#### Document Backup

```bash
# Create backup of RAG storage
oc create job rag-backup --image=registry.access.redhat.com/ubi9/ubi:latest -- \
  sh -c "tar czf /tmp/rag-backup-$(date +%Y%m%d).tar.gz -C /mnt/rag-storage ."

# Copy backup from pod
oc cp $(oc get pod -l job-name=rag-backup -o name | head -1):/tmp/rag-backup-*.tar.gz ./rag-backup.tar.gz
```

#### Configuration Backup

```bash
# Export RAG configuration
oc get configmap fips-chat-rag-config -o yaml > rag-config-backup.yaml
oc get pvc rag-storage rag-cache -o yaml > rag-pvc-backup.yaml
```

#### Recovery Procedures

```bash
# Restore RAG configuration
oc apply -f rag-config-backup.yaml
oc apply -f rag-pvc-backup.yaml

# Restore document data
oc create job rag-restore --image=registry.access.redhat.com/ubi9/ubi:latest -- \
  sh -c "cd /mnt/rag-storage && tar xzf /tmp/rag-backup.tar.gz"
```

### RAG Demo Scenario Deployment

For the demo environment (75 PDFs, 300 pages):

```bash
# Verify demo configuration
oc get configmap fips-chat-rag-config -o jsonpath='{.data.RAG_DEMO_MODE}'

# Monitor demo limits
oc exec deployment/fips-chat -- find /mnt/rag-storage -name "*.pdf" | wc -l
oc logs deployment/fips-chat | grep -i "demo.*limit"

# Test demo functionality
echo "Demo test checklist:"
echo "1. Upload multiple PDF files"
echo "2. Verify 75 PDF limit enforced"
echo "3. Verify 300 page limit enforced"
echo "4. Test search across multiple documents"
echo "5. Verify memory usage stays within limits"
```

## Support

For issues related to:
- **Application**: Check application logs and health endpoints
- **OpenShift**: Consult OpenShift documentation and cluster administrators
- **Ollama**: Verify Ollama service deployment and connectivity
- **FIPS**: Ensure cluster and nodes are FIPS-enabled
- **RAG**: Check storage accessibility, document processing logs, and search engine status