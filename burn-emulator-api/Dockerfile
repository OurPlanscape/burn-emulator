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

USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["/app/burn-emulator-api"]
