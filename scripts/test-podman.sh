#!/bin/bash
# Test script for Podman container

set -euo pipefail

# Configuration
IMAGE_NAME="ollama-streamlit"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CONTAINER_NAME="FIPS-Chat"
TEST_PORT="${TEST_PORT:-8080}"

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

cleanup() {
    log_info "Cleaning up..."
    if podman container exists "${CONTAINER_NAME}"; then
        podman stop "${CONTAINER_NAME}" 2>/dev/null || true
        podman rm "${CONTAINER_NAME}" 2>/dev/null || true
    fi
}

# Set up cleanup trap
trap cleanup EXIT

# Check if image exists
if ! podman image exists "${IMAGE_NAME}:${IMAGE_TAG}"; then
    log_error "Image ${IMAGE_NAME}:${IMAGE_TAG} not found. Build it first with:"
    echo "  ./scripts/build-podman.sh"
    exit 1
fi

# Clean up any existing container
cleanup

log_info "Starting test container..."

# Run the container in background
podman run -d \
    --name "${CONTAINER_NAME}" \
    --publish "${TEST_PORT}:8080" \
    --env OLLAMA_HOST="http://host.containers.internal:11434" \
    --env PYTHONUNBUFFERED=1 \
    --health-cmd "curl -f http://localhost:8080/_stcore/health || exit 1" \
    --health-interval 30s \
    --health-timeout 10s \
    --health-start-period 30s \
    --health-retries 3 \
    "${IMAGE_NAME}:${IMAGE_TAG}"

log_info "Container started. Waiting for application to be ready..."

# Wait for container to be healthy
TIMEOUT=120
ELAPSED=0
SLEEP_INTERVAL=5

while [[ $ELAPSED -lt $TIMEOUT ]]; do
    if podman healthcheck run "${CONTAINER_NAME}" &>/dev/null; then
        log_info "Container is healthy!"
        break
    fi
    
    if ! podman container exists "${CONTAINER_NAME}" || [[ "$(podman inspect "${CONTAINER_NAME}" --format '{{.State.Status}}')" != "running" ]]; then
        log_error "Container is not running!"
        log_info "Container logs:"
        podman logs "${CONTAINER_NAME}"
        exit 1
    fi
    
    log_info "Waiting for container to be ready... (${ELAPSED}s/${TIMEOUT}s)"
    sleep $SLEEP_INTERVAL
    ELAPSED=$((ELAPSED + SLEEP_INTERVAL))
done

if [[ $ELAPSED -ge $TIMEOUT ]]; then
    log_error "Container health check timed out after ${TIMEOUT}s"
    log_info "Container logs:"
    podman logs "${CONTAINER_NAME}"
    exit 1
fi

# Test HTTP endpoints
log_info "Testing HTTP endpoints..."

# Test health endpoint
log_info "Testing health endpoint..."
if curl -s -f "http://localhost:${TEST_PORT}/_stcore/health" > /dev/null; then
    log_info "✅ Health endpoint responding"
else
    log_error "❌ Health endpoint not responding"
    podman logs "${CONTAINER_NAME}" --tail 20
    exit 1
fi

# Test main application
log_info "Testing main application endpoint..."
if curl -s -f "http://localhost:${TEST_PORT}/" > /dev/null; then
    log_info "✅ Main application responding"
else
    log_warn "⚠️ Main application not responding (this might be expected for Streamlit)"
fi

# Show container info
log_info "Container information:"
echo "Container ID: $(podman inspect "${CONTAINER_NAME}" --format '{{.Id}}')"
echo "Image: $(podman inspect "${CONTAINER_NAME}" --format '{{.Config.Image}}')"
echo "Status: $(podman inspect "${CONTAINER_NAME}" --format '{{.State.Status}}')"
echo "Started: $(podman inspect "${CONTAINER_NAME}" --format '{{.State.StartedAt}}')"

# Show resource usage
log_info "Resource usage:"
podman stats "${CONTAINER_NAME}" --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}"

# Show recent logs
log_info "Recent logs (last 10 lines):"
podman logs "${CONTAINER_NAME}" --tail 10

# Interactive mode option
if [[ "${1:-}" == "--interactive" ]]; then
    log_info "Application is running at http://localhost:${TEST_PORT}"
    log_info "Press Ctrl+C to stop the container"
    
    # Follow logs
    podman logs -f "${CONTAINER_NAME}"
else
    log_info "Test completed successfully!"
    log_info "To run interactively: $0 --interactive"
    log_info "To connect manually: curl http://localhost:${TEST_PORT}/_stcore/health"
fi