# =============================================================================
# Project Aurelius - Dockerfile (Linux x86_64 with CUDA support)
# =============================================================================
#
# Build:
#   docker build -t aurelius:latest .
#
# Run:
#   docker run --rm -v $(pwd):/data -w /data aurelius:latest python -m aurelius
# =============================================================================

FROM python:3.11-slim-bookworm

# Install system dependencies needed for RDKit, PyTorch, build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gcc \
    g++ \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch (CPU + CUDA 12.1), torchvision, and MLX
# Using pip indexes to avoid conda-forge version conflicts
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install MLX (Apple Silicon ML framework) — available via pip
RUN pip install --no-cache-dir mlx==0.21.0

# Install RDKit (wheel available on PyPI for x86_64 Linux)
RUN pip install --no-cache-dir rdkit==2024.10.4

# Install remaining project dependencies
COPY requirements-full.txt /tmp/requirements-full.txt
RUN pip install --no-cache-dir -r /tmp/requirements-full.txt

# Copy project source
COPY . /app/aurelius
WORKDIR /app

# Default command
ENTRYPOINT ["python", "-m", "aurelius"]
