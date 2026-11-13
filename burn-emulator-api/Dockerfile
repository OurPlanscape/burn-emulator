# build context: repo root
FROM golang:1.26-trixie AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgeos-dev libproj-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY burn-emulator-api/go.mod burn-emulator-api/go.sum ./
RUN go mod download
COPY burn-emulator-api/ .
RUN CGO_ENABLED=1 GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/burn-emulator-api ./cmd/server

FROM debian:trixie-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgeos-c1t64 libproj25 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /out/burn-emulator-api /app/burn-emulator-api

# claims are queued for the runner's claim loop on prem (BURN_EMULATOR_LOOP=1) instead of triggering runner jobs
# TODO: add offloading functionality when GPU quotas are available (BURN_EMULATOR_LOOP=0)
ENV BURN_EMULATOR_METAL=0
# metal: PT runs trigger the CPU runner job (BURN_EMULATOR_RUNNER_CPU_JOB) instead of being queued
ENV BURN_EMULATOR_CPU_OFFLOAD=1

USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["/app/burn-emulator-api"]
