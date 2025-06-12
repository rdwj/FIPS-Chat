# OpenShift Deployment Files

This directory contains all the necessary Kubernetes/OpenShift manifests for deploying FIPS Chat.

## Files Overview

| File | Purpose | Description |
|------|---------|-------------|
| `configmap.yaml` | Configuration | Application settings and Streamlit config |
| `deployment.yaml` | Application | Main deployment with FIPS-compliant security context |
| `service.yaml` | Networking | ClusterIP service for internal communication |
| `route.yaml` | External Access | OpenShift route with TLS termination |
| `serviceaccount.yaml` | Security | Dedicated service account with minimal permissions |
| `networkpolicy.yaml` | Network Security | Restricted ingress/egress rules |
| `kustomization.yaml` | Orchestration | Kustomize configuration for deployment variants |

## Quick Deployment

```bash
# Deploy all resources with Kustomize
oc apply -k .

# Or deploy individually
oc apply -f serviceaccount.yaml
oc apply -f configmap.yaml
oc apply -f deployment.yaml
oc apply -f service.yaml
oc apply -f route.yaml
oc apply -f networkpolicy.yaml
```

## Configuration

### Required Updates Before Deployment

1. **Container Image**: Update image reference in `deployment.yaml` or `kustomization.yaml`
2. **Route Host**: Update hostname in `route.yaml` to match your cluster domain
3. **Ollama Service**: Ensure `OLLAMA_HOST` in `configmap.yaml` points to your Ollama service

### Environment-Specific Customization

Use Kustomize overlays for different environments:

```bash
# Create environment overlay
mkdir -p overlays/production
cat > overlays/production/kustomization.yaml << EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: ollama-production

resources:
- ../../

patchesStrategicMerge:
- replica-patch.yaml

images:
- name: fips-chat
  newName: your-registry.com/namespace/fips-chat
  newTag: v1.0.0
EOF

# Deploy with overlay
oc apply -k overlays/production/
```

## Security Features

- **FIPS Compliance**: OPENSSL_FIPS=1 environment variables
- **Non-root User**: Runs as UID 1001
- **Security Context**: Restrictive security context with dropped capabilities
- **Network Policy**: Limited ingress/egress traffic
- **Service Account**: Minimal permissions with automountServiceAccountToken=false

## Monitoring

The deployment includes comprehensive health checks:

- **Startup Probe**: 60s max startup time
- **Readiness Probe**: Traffic readiness check
- **Liveness Probe**: Application health monitoring

## Resources

Default resource allocations:

- **Requests**: 250m CPU, 512Mi memory
- **Limits**: 1000m CPU, 2Gi memory

Adjust based on your requirements and cluster capacity.