FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir rdkit==2024.09.6

RUN wget -q https://github.com/grimme-lab/xtb/releases/download/v6.6.1/xtb-6.6.1-linux-x86_64.tar.xz \
    && tar -xf xtb-6.6.1-linux-x86_64.tar.xz \
    && cp xtb-6.6.1/bin/xtb /usr/local/bin/ \
    && rm -rf xtb-6.6.1*

COPY . .
RUN pip install --no-cache-dir -e ".[web]"

RUN (command -v xtb >/dev/null 2>&1 && echo "xTB found") \
    || echo "WARNING: xTB binary not found. Falling back to TOM oracle."

EXPOSE 8000

ENTRYPOINT ["uvicorn", "server.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
