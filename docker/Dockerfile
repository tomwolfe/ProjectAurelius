FROM mambaorg/micromamba:1.5

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gcc \
    g++ \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN micromamba install -y -n base python=3.11 rdkit=2024.09.6 numpy scipy pandas scikit-learn joblib tqdm click structlog psutil

COPY requirements-full.txt /tmp/requirements-full.txt
RUN pip install -r /tmp/requirements-full.txt

COPY . /app/aurelius

ENTRYPOINT ["python", "-m", "aurelius"]
