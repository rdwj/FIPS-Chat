# OCI-compliant Containerfile for Chat Playground
# Multi-provider AI chat platform optimized for OpenShift with FIPS mode support

FROM registry.access.redhat.com/ubi9/python-311:latest

# Set environment variables for FIPS compliance
ENV OPENSSL_FIPS=1 \
    OPENSSL_FORCE_FIPS_MODE=1

# Container metadata following OCI spec
LABEL name="chat-playground-app" \
      version="1.0.0" \
      description="Multi-provider AI chat and image analysis platform" \
      maintainer="Development Team" \
      vendor="Organization" \
      io.k8s.description="Multi-provider AI chat platform supporting Ollama, vLLM, and other APIs" \
      io.k8s.display-name="Chat Playground" \
      io.openshift.tags="streamlit,ai,chat,vllm,ollama,openai,python"

# The UBI9 Python image already has a user with UID 1001, so we'll use it
USER 0

# Set working directory
WORKDIR /opt/app-root/src

# Copy requirements first for better layer caching
COPY --chown=1001:1001 requirements.txt ./

# Install Python dependencies with FIPS-compatible settings
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=1001:1001 . ./

# Create necessary directories with proper permissions
RUN mkdir -p /opt/app-root/src/.streamlit && \
    chown -R 1001:1001 /opt/app-root/src

# Switch to non-root user
USER 1001

# Create Streamlit config file for production  
RUN echo '[server]' > /opt/app-root/src/.streamlit/config.toml && \
    echo 'port = 8080' >> /opt/app-root/src/.streamlit/config.toml && \
    echo 'address = "0.0.0.0"' >> /opt/app-root/src/.streamlit/config.toml && \
    echo 'enableCORS = false' >> /opt/app-root/src/.streamlit/config.toml && \
    echo 'enableXsrfProtection = true' >> /opt/app-root/src/.streamlit/config.toml && \
    echo 'maxUploadSize = 10' >> /opt/app-root/src/.streamlit/config.toml && \
    echo '' >> /opt/app-root/src/.streamlit/config.toml && \
    echo '[browser]' >> /opt/app-root/src/.streamlit/config.toml && \
    echo 'gatherUsageStats = false' >> /opt/app-root/src/.streamlit/config.toml && \
    echo 'showErrorDetails = false' >> /opt/app-root/src/.streamlit/config.toml && \
    echo '' >> /opt/app-root/src/.streamlit/config.toml && \
    echo '[logger]' >> /opt/app-root/src/.streamlit/config.toml && \
    echo 'level = "info"' >> /opt/app-root/src/.streamlit/config.toml && \
    echo '' >> /opt/app-root/src/.streamlit/config.toml && \
    echo '[client]' >> /opt/app-root/src/.streamlit/config.toml && \
    echo 'toolbarMode = "minimal"' >> /opt/app-root/src/.streamlit/config.toml

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/_stcore/health || exit 1

# Expose port
EXPOSE 8080

# Set entrypoint and command
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true"]