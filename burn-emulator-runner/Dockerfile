FROM python:3.12-slim-bookworm

# BURN_EMULATOR_LOOP=1 (slurm): watch and run queued claims instead of one execution's request
# change RUN_DTYPE bfloat16 for faster inference when switched back to GPU
# RUN_DEVICE the Cloud Run CPU job sets cpu
ENV PYTHONUNBUFFERED=1 \
    BURN_EMULATOR_LOOP=0 \
    USE_CLOUD_PATHS=1 \
    RUN_DEVICE=cuda \
    RUN_DTYPE=float32

RUN mkdir -p /models

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libexpat1 libgeos-c1v5 build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY burn-emulator-model/ ./burn-emulator-model/
COPY burn-emulator-runner/ ./burn-emulator-runner/

WORKDIR /app/burn-emulator-runner
RUN uv sync --locked --no-cache

ENTRYPOINT ["uv", "run", "--no-sync", "burn-emulator-runner"]
