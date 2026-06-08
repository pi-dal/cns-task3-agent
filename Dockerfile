FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Base system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    zip \
    && rm -rf /var/lib/apt/lists/*

# Install torch with CUDA 12.4 support + pdm in one layer
RUN pip3 install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 \
    && pip3 install --no-cache-dir pdm

WORKDIR /app

# Copy dependency manifests
COPY pyproject.toml pdm.lock run.sh ./

# Install prod deps (torch already in system site-packages, pdm will reuse it)
RUN pdm install --prod --frozen-lockfile

# Copy source code (changes most often → keep last for layer cache)
COPY configs/ ./configs/
COPY src/ ./src/

# Clean up caches
RUN rm -rf /root/.cache /tmp/* /app/.venv/src

ENV PYTHONPATH=/app
ENV INPUT_DIR=/saisdata
ENV OUTPUT_DIR=/saisresult

ENTRYPOINT ["bash", "run.sh"]
