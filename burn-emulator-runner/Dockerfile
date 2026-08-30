FROM python:3.12-slim-bookwor

ENV PYTHONUNBUFFERED=1 \
    USE_CLOUD_PATHS=1 \
    RUN_DEVICE=cuda \
    RUN_DTYPE=bfloat16 \
    BURN_EMULATOR_MODELS_DIR=/models \
    PORT=8080

RUN mkdir -p /models

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libexpat1 libgeos-c1v5 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY burn-emulator-model/ ./burn-emulator-model/
COPY burn-emulator-runner/ ./burn-emulator-runner/

WORKDIR /app/burn-emulator-runner
RUN uv sync --no-cache

EXPOSE 8080
ENTRYPOINT ["uv", "run", "--no-sync", "burn-emulator-runner"]
