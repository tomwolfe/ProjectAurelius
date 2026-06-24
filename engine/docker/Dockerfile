# =============================================================================
# Stage 1 — Build dependencies
# =============================================================================
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# System build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install RDKit (C++ bindings require pip install)
RUN pip install --no-cache-dir rdkit==2024.09.6

# Install runtime dependencies
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir -e ".[api]"


# =============================================================================
# Stage 2 — Runtime image
# =============================================================================
FROM python:3.11-slim-bookworm

WORKDIR /app

# Runtime system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source (for entry_points, scripts, etc.)
COPY --from=builder /build/src /app/src

# Graceful xtb handling: warn if binary is missing, never crash
RUN (command -v xtb >/dev/null 2>&1 && echo "xTB found") \
    || echo "WARNING: xTB binary not found. Falling back to TOM oracle."

EXPOSE 8000

ENTRYPOINT ["uvicorn", "aurelius.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
