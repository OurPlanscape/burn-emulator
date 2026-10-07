# build context: repo root
FROM golang:1.26-alpine AS builder

WORKDIR /src
COPY burn-emulator-api/go.mod burn-emulator-api/go.sum ./
RUN go mod download
COPY burn-emulator-api/ .
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/burn-emulator-api ./cmd/server

FROM gcr.io/distroless/static-debian12:nonroot
WORKDIR /app
COPY --from=builder /out/burn-emulator-api /app/burn-emulator-api

EXPOSE 8080
ENTRYPOINT ["/app/burn-emulator-api"]
