FROM mambaorg/micromamba:1.5

# Install system dependencies needed for RDKit
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gcc \
    g++ \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create app directory and set working directory
WORKDIR /app

# Install project dependencies using conda first, then pip
COPY requirements-full.txt /tmp/requirements-full.txt
RUN micromamba install -y -n base --file /tmp/requirements-full.txt

# Install RDKit specifically from conda-forge
RUN micromamba install -y -n base rdkit=2024.09.6

# Copy project source
COPY . /app/aurelius

# Default command
ENTRYPOINT ["python", "-m", "aurelius"]
