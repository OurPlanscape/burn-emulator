package main

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"time"

	"burn-emulator-api/internal/dispatch"
	"burn-emulator-api/internal/handlers"
)

// read a required env var, or exit.
func env(key string) string {
	v := os.Getenv(key)
	if v == "" {
		slog.Error("missing required env var", "key", key)
		os.Exit(1)
	}
	return v
}

// read an optional env var, or return fallback.
func envDefault(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// wire up the dispatch client and serve POST /v1/jobs and GET /healthz on :8080.
func main() {
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, nil)))

	ctx := context.Background()

	cfg := dispatch.Config{
		ModelsURI:    env("BURN_EMULATOR_MODELS_URI"),
		OutputBucket: env("BURN_EMULATOR_OUTPUT_BUCKET"),
		RunnerURL:    env("BURN_EMULATOR_RUNNER_URL"),
	}

	client, err := dispatch.NewClient(ctx, cfg)
	if err != nil {
		slog.Error("failed to init dispatch client", "error", err)
		os.Exit(1)
	}

	varLocsPath := envDefault("VARLOCS_FILE", "configs/varlocs.yaml")
	validVarLocs, err := handlers.LoadVarLocs(varLocsPath)
	if err != nil {
		slog.Error("failed to load varlocs allow-list", "path", varLocsPath, "error", err)
		os.Exit(1)
	}

	jobsHandler := &handlers.JobsHandler{
		Dispatch: client,
		VarLocs:  validVarLocs,
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
		ReadTimeout:       15 * time.Second,
		// a request blocks on the synchronous runner call; keep this above the
		// handler's own request timeout.
		WriteTimeout: 150 * time.Second,
	}

	slog.Info("burn-emulator-api listening", "addr", srv.Addr)
	if err := srv.ListenAndServe(); err != nil {
		slog.Error("server exited", "error", err)
		os.Exit(1)
	}
}
