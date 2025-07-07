#!/bin/bash

# FIPS Chat RAG OpenShift Deployment Script
# This script automates the deployment of FIPS Chat with RAG functionality to OpenShift

set -e  # Exit on any error

# Configuration
NAMESPACE="${NAMESPACE:-ollama-platform}"
IMAGE_REGISTRY="${IMAGE_REGISTRY:-quay.io/wjackson}"
IMAGE_NAME="${IMAGE_NAME:-fips-chat}"
IMAGE_TAG="${IMAGE_TAG:-rag-v1.0}"
FULL_IMAGE="${IMAGE_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if oc is installed
    if ! command -v oc &> /dev/null; then
        log_error "OpenShift CLI (oc) is not installed"
        exit 1
    fi
    
    # Check if logged in to OpenShift
    if ! oc whoami &> /dev/null; then
        log_error "Not logged in to OpenShift. Please run 'oc login' first"
        exit 1
    fi
    
    # Check if podman is available for building
    if ! command -v podman &> /dev/null; then
        log_warning "Podman not found. You may need to build the container image separately"
    fi
    
    log_success "Prerequisites check completed"
}

# Create or switch to namespace
setup_namespace() {
    log_info "Setting up namespace: $NAMESPACE"
    
    if oc get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Namespace $NAMESPACE already exists"
        oc project "$NAMESPACE"
    else
        log_info "Creating namespace: $NAMESPACE"
        oc new-project "$NAMESPACE"
    fi
    
    log_success "Namespace setup completed"
}

# Build and push container image
build_and_push_image() {
    if [ "$SKIP_BUILD" = "true" ]; then
        log_info "Skipping image build (SKIP_BUILD=true)"
        return
    fi
    
    log_info "Building container image: $FULL_IMAGE"
    
    if ! command -v podman &> /dev/null; then
        log_error "Podman not found. Please build the image manually or set SKIP_BUILD=true"
        exit 1
    fi
    
    # Build image
    podman build -f Containerfile -t "$FULL_IMAGE" .
    
    # Push image
    log_info "Pushing image to registry: $FULL_IMAGE"
    podman push "$FULL_IMAGE"
    
    log_success "Image build and push completed"
}

# Update deployment configuration
update_deployment_config() {
    log_info "Updating deployment configuration with image: $FULL_IMAGE"
    
    # Update image reference in deployment.yaml
    sed -i.bak "s|image:.*|image: $FULL_IMAGE|" openshift/deployment.yaml
    
    log_success "Deployment configuration updated"
}

# Deploy RAG storage components
deploy_storage() {
    log_info "Deploying RAG storage components..."
    
    # Deploy persistent volume claims
    oc apply -f openshift/rag-pvc.yaml
    
    # Wait for PVCs to be bound
    log_info "Waiting for persistent volumes to be bound..."
    oc wait --for=condition=Bound pvc/rag-storage --timeout=300s
    oc wait --for=condition=Bound pvc/rag-cache --timeout=300s
    
    log_success "RAG storage components deployed"
}

# Deploy configuration
deploy_config() {
    log_info "Deploying configuration..."
    
    # Deploy base configuration
    oc apply -f openshift/configmap.yaml
    
    # Deploy RAG-specific configuration
    oc apply -f openshift/rag-configmap.yaml
    
    log_success "Configuration deployed"
}

# Deploy application
deploy_application() {
    log_info "Deploying application..."
    
    # Deploy service account
    oc apply -f openshift/serviceaccount.yaml
    
    # Deploy application
    oc apply -f openshift/deployment.yaml
    oc apply -f openshift/service.yaml
    oc apply -f openshift/route.yaml
    
    # Deploy network policy
    oc apply -f openshift/networkpolicy.yaml
    
    log_success "Application deployed"
}

# Wait for deployment to be ready
wait_for_deployment() {
    log_info "Waiting for deployment to be ready..."
    
    # Wait for deployment rollout
    oc rollout status deployment/fips-chat --timeout=600s
    
    # Wait for pods to be ready
    oc wait --for=condition=Ready pod -l app=fips-chat --timeout=300s
    
    log_success "Deployment is ready"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check pod status
    echo "Pod status:"
    oc get pods -l app=fips-chat
    echo
    
    # Check services
    echo "Services:"
    oc get svc -l app=fips-chat
    echo
    
    # Check routes
    echo "Routes:"
    oc get route -l app=fips-chat
    echo
    
    # Check PVCs
    echo "Persistent Volume Claims:"
    oc get pvc
    echo
    
    # Check health endpoint
    log_info "Checking application health..."
    
    # Get pod name
    POD_NAME=$(oc get pod -l app=fips-chat -o jsonpath='{.items[0].metadata.name}')
    
    if [ -n "$POD_NAME" ]; then
        # Check basic health
        if oc exec "$POD_NAME" -- curl -f -s http://localhost:8080/_stcore/health > /dev/null; then
            log_success "Application health check passed"
        else
            log_warning "Application health check failed"
        fi
        
        # Check RAG health
        log_info "Checking RAG system health..."
        oc exec "$POD_NAME" -- curl -s http://localhost:8080/_stcore/health 2>/dev/null | grep -q rag_system && \
            log_success "RAG system health check passed" || \
            log_warning "RAG system health check not available"
        
        # Check storage mounts
        log_info "Checking storage mounts..."
        if oc exec "$POD_NAME" -- test -d /mnt/rag-storage; then
            log_success "RAG storage mount verified"
        else
            log_error "RAG storage mount not found"
        fi
        
        if oc exec "$POD_NAME" -- test -d /mnt/rag-cache; then
            log_success "RAG cache mount verified"
        else
            log_error "RAG cache mount not found"
        fi
    else
        log_error "Could not find application pod"
    fi
    
    # Get application URL
    APP_URL=$(oc get route fips-chat -o jsonpath='{.spec.host}' 2>/dev/null || echo "Route not found")
    if [ "$APP_URL" != "Route not found" ]; then
        log_success "Application available at: https://$APP_URL"
        echo
        echo "RAG Testing Instructions:"
        echo "1. Navigate to https://$APP_URL"
        echo "2. Go to the 'Documents' tab"
        echo "3. Upload a PDF document"
        echo "4. Wait for processing to complete"
        echo "5. Test RAG-enhanced chat in the 'Chat' tab"
    fi
}

# Main deployment function
main() {
    echo "================================================"
    echo "FIPS Chat RAG OpenShift Deployment"
    echo "================================================"
    echo "Namespace: $NAMESPACE"
    echo "Image: $FULL_IMAGE"
    echo "================================================"
    echo
    
    check_prerequisites
    setup_namespace
    build_and_push_image
    update_deployment_config
    deploy_storage
    deploy_config
    deploy_application
    wait_for_deployment
    verify_deployment
    
    echo
    echo "================================================"
    log_success "RAG deployment completed successfully!"
    echo "================================================"
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "FIPS Chat RAG OpenShift Deployment Script"
        echo
        echo "Usage: $0 [OPTIONS]"
        echo
        echo "Environment Variables:"
        echo "  NAMESPACE        OpenShift namespace (default: ollama-platform)"
        echo "  IMAGE_REGISTRY   Container registry (default: quay.io/wjackson)"
        echo "  IMAGE_NAME       Image name (default: fips-chat)"
        echo "  IMAGE_TAG        Image tag (default: rag-v1.0)"
        echo "  SKIP_BUILD       Skip image build (default: false)"
        echo
        echo "Examples:"
        echo "  $0                                    # Deploy with defaults"
        echo "  NAMESPACE=my-project $0               # Deploy to custom namespace"
        echo "  SKIP_BUILD=true $0                    # Skip image build"
        echo "  IMAGE_TAG=v2.0 $0                     # Use custom image tag"
        echo
        exit 0
        ;;
    --verify|-v)
        log_info "Running verification only..."
        verify_deployment
        exit 0
        ;;
    "")
        main
        ;;
    *)
        log_error "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac 