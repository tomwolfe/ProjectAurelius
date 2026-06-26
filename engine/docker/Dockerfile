FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir rdkit==2024.09.6

COPY . .
RUN pip install --no-cache-dir -e ".[api]"

RUN (command -v xtb >/dev/null 2>&1 && echo "xTB found") \
    || echo "WARNING: xTB binary not found. Falling back to TOM oracle."

EXPOSE 8000

ENTRYPOINT ["uvicorn", "server.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
