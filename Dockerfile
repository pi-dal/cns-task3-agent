FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS base

# Base system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    zip \
    && rm -rf /var/lib/apt/lists/*

# Install torch with CUDA support — skip bundled NVIDIA libs (provided by base image)
# The --find-links pre-resolves all dependencies so pip doesn't re-resolve
RUN pip3 install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 \
    && pip3 install --no-cache-dir pdm

WORKDIR /app

COPY pyproject.toml pdm.lock run.sh ./
COPY configs/ ./configs/
COPY src/ ./src/
COPY tests/ ./tests/

# Install only prod deps
RUN pdm install --prod --no-lock

# Clean pdm cache and pip cache to reduce image size
RUN rm -rf /root/.cache /tmp/* /app/.venv/src

ENV PYTHONPATH=/app
ENV INPUT_DIR=/saisdata
ENV OUTPUT_DIR=/saisresult

ENTRYPOINT ["bash", "run.sh"]
