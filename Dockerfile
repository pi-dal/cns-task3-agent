FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    zip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir pdm

WORKDIR /app

COPY pyproject.toml pdm.lock run.sh ./
COPY configs/ ./configs/
COPY src/ ./src/
COPY tests/ ./tests/

RUN pdm install --prod --no-lock

ENV PYTHONPATH=/app
ENV INPUT_DIR=/saisdata
ENV OUTPUT_DIR=/saisresult

ENTRYPOINT ["bash", "run.sh"]
