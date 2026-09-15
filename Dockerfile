# Multi-Stage Production Dockerfile for Sentinel Gujarat (80,000 Camera Ready)
# Base Image: NVIDIA CUDA 12.2 Runtime with Ubuntu 22.04
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install OS runtime dependencies & FFmpeg with NVDEC support
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create unprivileged application user
RUN useradd -m -u 10001 -s /bin/bash sentinel && \
    mkdir -p /app/data /app/output /app/logs && \
    chown -R sentinel:sentinel /app

# Install Python requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source code
COPY --chown=sentinel:sentinel . /app

USER sentinel
EXPOSE 8000

# Health check probe
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Launch production server with Uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--loop", "uvloop", "--http", "httptools"]
