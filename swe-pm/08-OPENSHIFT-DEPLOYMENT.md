# Chat Guide 08: OpenShift Deployment

## Objective
Create comprehensive OpenShift deployment configuration and documentation for the RAG-enabled FIPS Chat application, including persistent storage, configuration management, and production-ready deployment procedures.

## Prerequisites
- Chat Guide 01-07 completed (full RAG system tested and validated)
- All RAG functionality working and tested
- FIPS compliance validated
- Feature branch: `feature/rag-implementation`

## Goals for This Chat Session
1. Create OpenShift persistent volume configurations for RAG storage
2. Update deployment manifests for RAG functionality
3. Create ConfigMaps and Secrets for RAG configuration
4. Build production-ready container image with RAG dependencies
5. Create comprehensive deployment documentation
6. Add monitoring and troubleshooting guides

## Tasks to Complete

### 1. Persistent Storage Configuration
**What to Ask For:**
```
Create OpenShift persistent volume configuration:
- openshift/rag-pvc.yaml for RAG storage
- Configure appropriate storage class and size
- Set up proper access modes and retention policies
- Create backup and recovery procedures
- Add storage monitoring and alerts
```

### 2. ConfigMap and Environment Configuration
**What to Ask For:**
```
Create RAG-specific configuration management:
- openshift/rag-configmap.yaml with all RAG settings
- Update existing configmap.yaml to include RAG variables
- Create production-appropriate configuration values
- Add FIPS compliance settings
- Document all configuration options
```

### 3. Deployment Manifest Updates
**What to Ask For:**
```
Update openshift/deployment.yaml for RAG:
- Add RAG storage volume mounts
- Update resource limits for RAG workloads
- Add RAG environment variables
- Update health checks to include RAG functionality
- Configure FIPS compliance settings
```

### 4. Container Image Build
**What to Ask For:**
```
Update container build for RAG dependencies:
- Add RAG dependencies to requirements.txt
- Update Containerfile for FIPS-compliant RAG build
- Add Docling and scikit-learn to container
- Verify FIPS compliance in container build
- Create container build and push scripts
```

### 5. Network and Security Configuration
**What to Ask For:**
```
Update security and networking for RAG:
- Update networkpolicy.yaml for RAG traffic
- Configure security contexts for file storage
- Add RBAC permissions if needed
- Configure encryption at rest for storage
- Document security considerations
```

### 6. Monitoring and Observability
**What to Ask For:**
```
Add monitoring for RAG functionality:
- Health check endpoints for RAG components
- Metrics collection for document processing
- Storage usage monitoring
- Performance metrics collection
- Alert configurations for RAG issues
```

### 7. Deployment Documentation
**What to Ask For:**
```
Create comprehensive deployment guide:
- Update DEPLOYMENT.md with RAG procedures
- Step-by-step RAG deployment instructions
- Configuration reference documentation
- Troubleshooting guide for common issues
- Backup and recovery procedures
- Performance tuning guidance
```

## Expected Outputs After This Chat
- [ ] Complete OpenShift manifests for RAG deployment
- [ ] Persistent volume configuration for RAG storage
- [ ] Updated container build with RAG dependencies
- [ ] ConfigMaps and environment configuration
- [ ] Security and networking configuration
- [ ] Monitoring and health check setup
- [ ] Comprehensive deployment documentation

## Key Implementation Details

### Persistent Volume Configuration
```yaml
# openshift/rag-pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: rag-storage
  labels:
    app: fips-chat
    component: rag-storage
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi  # Sufficient for demo (300 pages, 75 PDFs)
  storageClassName: gp3-csi  # Use appropriate storage class
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: rag-cache
  labels:
    app: fips-chat
    component: rag-cache
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 2Gi  # For search indexes and cache
  storageClassName: gp3-csi
```

### RAG ConfigMap
```yaml
# openshift/rag-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fips-chat-rag-config
  labels:
    app: fips-chat
    component: rag-config
data:
  # RAG Core Settings
  ENABLE_RAG: "true"
  RAG_STORAGE_PATH: "/mnt/rag-storage"
  RAG_CACHE_PATH: "/mnt/rag-cache"
  
  # Memory and Performance
  RAG_MAX_MEMORY_MB: "100"
  RAG_CACHE_SIZE: "50"
  
  # Document Processing
  RAG_CHUNK_SIZE: "1000"
  RAG_CHUNK_OVERLAP: "200"
  RAG_MAX_DOCUMENT_SIZE_MB: "10"
  RAG_MAX_DOCUMENTS: "100"
  
  # Search Configuration
  RAG_MAX_SEARCH_RESULTS: "5"
  RAG_RELEVANCE_THRESHOLD: "0.1"
  RAG_TFIDF_MAX_FEATURES: "5000"
  
  # Demo Environment
  RAG_DEMO_MODE: "true"
  RAG_DEMO_MAX_PAGES: "300"
  RAG_DEMO_MAX_PDFS: "75"
  
  # FIPS Compliance
  RAG_HASH_ALGORITHM: "sha256"
  OPENSSL_FIPS: "1"
  OPENSSL_FORCE_FIPS_MODE: "1"
```

### Updated Deployment Configuration
```yaml
# Addition to openshift/deployment.yaml
spec:
  template:
    spec:
      containers:
      - name: streamlit-app
        # ... existing configuration ...
        
        # Environment variables from RAG ConfigMap
        envFrom:
        - configMapRef:
            name: fips-chat-config
        - configMapRef:
            name: fips-chat-rag-config
        
        # Volume mounts for RAG storage
        volumeMounts:
        - name: streamlit-config
          mountPath: /opt/app-root/src/.streamlit
          readOnly: true
        - name: tmp-volume
          mountPath: /tmp
        - name: rag-storage
          mountPath: /mnt/rag-storage
        - name: rag-cache
          mountPath: /mnt/rag-cache
        
        # Updated resource limits for RAG
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"     # Increased for RAG processing
            cpu: "1000m"
        
        # Enhanced health checks
        livenessProbe:
          httpGet:
            path: /_stcore/health
            port: 8080
          initialDelaySeconds: 60   # Longer for RAG initialization
          periodSeconds: 30
          
        readinessProbe:
          httpGet:
            path: /_stcore/health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 10
      
      # Additional volumes
      volumes:
      - name: streamlit-config
        configMap:
          name: fips-chat-config
      - name: tmp-volume
        emptyDir:
          sizeLimit: 1Gi
      - name: rag-storage
        persistentVolumeClaim:
          claimName: rag-storage
      - name: rag-cache
        persistentVolumeClaim:
          claimName: rag-cache
```

### Container Build Updates
```dockerfile
# Updates to Containerfile
FROM registry.access.redhat.com/ubi9/python-39:latest

# Set FIPS mode
ENV OPENSSL_FIPS=1
ENV OPENSSL_FORCE_FIPS_MODE=1

# Install system dependencies for RAG
USER root
RUN dnf update -y && \
    dnf install -y \
        gcc \
        g++ \
        python39-devel \
        libffi-devel \
        openssl-devel && \
    dnf clean all

USER 1001

# Copy requirements with RAG dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create RAG storage directories
RUN mkdir -p /opt/app-root/src/rag_storage && \
    chmod 755 /opt/app-root/src/rag_storage

# Validate FIPS compliance
RUN python -c "import hashlib; print('FIPS validation:', hashlib.md5 if not hashlib.sha256 else 'SHA-256 OK')"

EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
```

### Security and Network Configuration
```yaml
# Updates to openshift/networkpolicy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: fips-chat-netpol
spec:
  podSelector:
    matchLabels:
      app: fips-chat
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: openshift-ingress
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS
    - protocol: TCP
      port: 53   # DNS
    - protocol: UDP
      port: 53   # DNS
  # No additional egress needed for RAG (file-based storage)
```

### Health Check Enhancement
```python
# Addition to health.py for RAG health checks
def check_rag_health():
    """Check RAG system health"""
    try:
        from rag.file_storage import FileStorage
        from rag.search_engine import TFIDFSearchEngine
        import os
        
        health_status = {
            "rag_enabled": os.getenv("ENABLE_RAG", "false").lower() == "true",
            "storage_accessible": False,
            "search_engine_ready": False,
            "fips_compliant": False
        }
        
        # Check storage accessibility
        storage_path = os.getenv("RAG_STORAGE_PATH", "./rag_storage")
        if os.path.exists(storage_path) and os.access(storage_path, os.W_OK):
            health_status["storage_accessible"] = True
        
        # Check FIPS compliance
        if os.getenv("OPENSSL_FIPS") == "1":
            health_status["fips_compliant"] = True
        
        # Check search engine initialization
        try:
            storage = FileStorage(storage_path)
            search_engine = TFIDFSearchEngine(storage)
            health_status["search_engine_ready"] = True
        except Exception:
            pass
        
        return health_status
        
    except Exception as e:
        return {"error": str(e)}
```

## Deployment Procedures

### Step-by-Step Deployment
```bash
# 1. Create feature branch and prepare code
git checkout -b feature/rag-implementation
git push -u origin feature/rag-implementation

# 2. Build and push container image
./scripts/build-podman.sh
podman tag fips-chat:latest quay.io/wjackson/fips-chat:rag-v1.0
podman push quay.io/wjackson/fips-chat:rag-v1.0

# 3. Update image reference in deployment
sed -i 's|fips-chat:latest|quay.io/wjackson/fips-chat:rag-v1.0|' openshift/deployment.yaml

# 4. Deploy RAG components
oc apply -f openshift/rag-pvc.yaml
oc apply -f openshift/rag-configmap.yaml
oc apply -f openshift/deployment.yaml

# 5. Verify deployment
oc get pods -l app=fips-chat
oc logs -l app=fips-chat -f

# 6. Test RAG functionality
oc port-forward service/fips-chat 8080:8080
# Test document upload and RAG queries
```

### Monitoring and Troubleshooting
```bash
# Check RAG storage
oc exec deployment/fips-chat -- ls -la /mnt/rag-storage

# Check RAG configuration
oc exec deployment/fips-chat -- env | grep RAG

# Monitor resource usage
oc top pods -l app=fips-chat

# Check FIPS compliance
oc exec deployment/fips-chat -- python -c "import os; print('FIPS:', os.getenv('OPENSSL_FIPS'))"

# View RAG health status
oc exec deployment/fips-chat -- curl localhost:8080/health
```

## Documentation Updates

### DEPLOYMENT.md Updates
```markdown
## RAG Functionality Deployment

### Prerequisites for RAG
- OpenShift cluster with persistent storage support
- FIPS-enabled nodes (for production)
- Minimum 10Gi storage for RAG data
- 2Gi memory limit for RAG processing

### RAG-Specific Deployment Steps

1. **Deploy Storage Components**
   ```bash
   oc apply -f openshift/rag-pvc.yaml
   ```

2. **Configure RAG Settings**
   ```bash
   oc apply -f openshift/rag-configmap.yaml
   ```

3. **Deploy Updated Application**
   ```bash
   oc apply -f openshift/deployment.yaml
   ```

4. **Verify RAG Functionality**
   - Navigate to the Documents tab
   - Upload a test PDF
   - Verify processing completes
   - Test RAG-enhanced chat

### RAG Configuration Reference
[Document all configuration options and their effects]

### Troubleshooting RAG Issues
[Common issues and solutions]
```

## Success Criteria for This Chat
- ✅ Complete OpenShift deployment configuration for RAG
- ✅ Persistent storage properly configured and accessible
- ✅ All RAG functionality works in deployed environment
- ✅ FIPS compliance maintained in OpenShift deployment
- ✅ Monitoring and health checks functional
- ✅ Comprehensive deployment documentation created
- ✅ Demo scenario (75 PDFs, 300 pages) deployable and functional

## Production Readiness Checklist
- [ ] Resource limits appropriate for expected load
- [ ] Storage backup and recovery procedures documented
- [ ] Security contexts and network policies configured
- [ ] Monitoring and alerting configured
- [ ] Log aggregation and analysis setup
- [ ] Disaster recovery procedures documented
- [ ] Performance baselines established

## Next Steps After Completion
1. Merge feature branch to main after successful deployment
2. Create release notes documenting RAG functionality
3. Update user documentation with RAG features
4. Plan training for operations teams
5. Schedule regular backup and maintenance procedures

## Notes for Implementation
- Test deployment in non-production environment first
- Validate FIPS compliance in actual OpenShift FIPS environment
- Monitor resource usage during initial deployment
- Document any deployment-specific issues and solutions
- Create rollback procedures in case of deployment issues
- Plan for future scaling and performance optimization