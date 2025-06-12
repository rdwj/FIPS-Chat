#!/bin/bash
# Build script for Podman (OCI-compliant)

set -euo pipefail

# Configuration
IMAGE_NAME="ollama-streamlit"
IMAGE_TAG="${IMAGE_TAG:-latest}"
BUILD_CONTEXT="${BUILD_CONTEXT:-.}"
CONTAINERFILE="${CONTAINERFILE:-Containerfile}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Pre-build checks
log_info "Starting Podman build for ${IMAGE_NAME}:${IMAGE_TAG}"

# Check if Podman is installed
if ! command -v podman &> /dev/null; then
    log_error "Podman is not installed or not in PATH"
    exit 1
fi

# Check if Containerfile exists
if [[ ! -f "${BUILD_CONTEXT}/${CONTAINERFILE}" ]]; then
    log_error "Containerfile not found at ${BUILD_CONTEXT}/${CONTAINERFILE}"
    exit 1
fi

# Check if we're in the right directory
if [[ ! -f "${BUILD_CONTEXT}/app.py" ]]; then
    log_error "app.py not found. Make sure you're in the project root directory."
    exit 1
fi

# Build the image
log_info "Building image with Podman..."
log_info "Build context: ${BUILD_CONTEXT}"
log_info "Containerfile: ${CONTAINERFILE}"

podman build \
    --file "${BUILD_CONTEXT}/${CONTAINERFILE}" \
    --tag "${IMAGE_NAME}:${IMAGE_TAG}" \
    --tag "${IMAGE_NAME}:latest" \
    --format oci \
    --layers \
    --pull \
    "${BUILD_CONTEXT}"

# Verify the build
if podman image exists "${IMAGE_NAME}:${IMAGE_TAG}"; then
    log_info "Build successful!"
    
    # Show image details
    log_info "Image details:"
    SIZE=$(podman image inspect "${IMAGE_NAME}:${IMAGE_TAG}" --format '{{.Size}}')
    echo "Size: $SIZE bytes"
    podman image inspect "${IMAGE_NAME}:${IMAGE_TAG}" --format '{{.Created}}' | xargs -I {} echo "Created: {}"
    
    # List layers
    log_info "Image layers:"
    podman image history "${IMAGE_NAME}:${IMAGE_TAG}" --format "table {{.Created}}\t{{.Size}}\t{{.CreatedBy}}" | head -10
    
else
    log_error "Build failed!"
    exit 1
fi

log_info "Build completed successfully!"
log_info "To run the container locally:"
echo "  podman run -p 8080:8080 --rm --name ollama-FIPS-Chat ${IMAGE_NAME}:${IMAGE_TAG}"
log_info "To push to a registry:"
echo "  podman tag ${IMAGE_NAME}:${IMAGE_TAG} your-registry.com/namespace/${IMAGE_NAME}:${IMAGE_TAG}"
echo "  podman push your-registry.com/namespace/${IMAGE_NAME}:${IMAGE_TAG}"