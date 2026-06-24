FROM python:3.11-slim-bookworm

# Install system dependencies needed for RDKit
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gcc \
    g++ \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install RDKit
RUN pip install --no-cache-dir rdkit==2024.09.6

# Install remaining project dependencies
COPY requirements-full.txt /tmp/requirements-full.txt
RUN pip install --no-cache-dir -r /tmp/requirements-full.txt

# Install API server dependencies
RUN pip install --no-cache-dir fastapi uvicorn diskcache

# Copy project source
COPY . /app/aurelius
WORKDIR /app

# Default command — start uvicorn API server
ENTRYPOINT ["uvicorn", "aurelius.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
