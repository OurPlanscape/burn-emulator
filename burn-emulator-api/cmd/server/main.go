package main

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"

	"burn-emulator-api/internal/auth"
	"burn-emulator-api/internal/gke"
	"burn-emulator-api/internal/handlers"
	"burn-emulator-api/internal/rate_limit"
)

func env(key string) string {
	v := os.Getenv(key)
	if v == "" {
		slog.Error("missing required env var", "key", key)
		os.Exit(1)
	}
	return v
}

func envDefault(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func main() {
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, nil)))

	ctx := context.Background()

	cfg := k8s.Config{
		Namespace:      env("BURN_EMULATOR_JOB_NAMESPACE"),
		ServiceAccount: env("BURN_EMULATOR_JOB_SERVICE_ACCOUNT"),
		ImageStore:     env("BURN_EMULATOR_ARTIFACT_STORE"),
		OutputBucket:   env("BURN_EMULATOR_OUTPUT_BUCKET"),
		RedisAddr:      env("BURN_EMULATOR_REDIS_ADDR"),
	}

	k8sClient, err := k8s.NewClient(ctx, cfg)
	if err != nil {
		slog.Error("failed to init k8s client", "error", err)
		os.Exit(1)
	}

	variationsPath := envDefault("VARIATIONS_FILE", "configs/variations.yaml")
	validVariations, err := handlers.LoadVariations(variationsPath)
	if err != nil {
		slog.Error("failed to load variations allow-list", "path", variationsPath, "error", err)
		os.Exit(1)
	}

	allowedCallers := strings.Split(env("BURN_EMULATOR_ALLOWED_CALLERS"), ",")
	verifier, err := auth.NewVerifier(ctx, env("BURN_EMULATOR_TOKEN_AUDIENCE"), allowedCallers)
	if err != nil {
		slog.Error("failed to init auth verifier", "error", err)
		os.Exit(1)
	}
	limiter := ratelimit.NewStore(1, 5)

	stop := make(chan struct{})
	defer close(stop)
	go limiter.Cleanup(10*time.Minute, stop)

	jobsHandler := &handlers.JobsHandler{
		K8s:        k8sClient,
		Verifier:   verifier,
		Limiter:    limiter,
		Variations: validVariations,
	}

	mux := http.NewServeMux()
	mux.Handle("/v1/jobs", jobsHandler)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("ok"))
	})

	srv := &http.Server{
		Addr:              ":8080",
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      10 * time.Second,
	}

	slog.Info("burn-emulator-api listening", "addr", srv.Addr)
	if err := srv.ListenAndServe(); err != nil {
		slog.Error("server exited", "error", err)
		os.Exit(1)
	}
}