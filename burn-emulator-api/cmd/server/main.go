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

func env(key string) string {
	v := os.Getenv(key)
	if v == "" {
		slog.Error("missing required env var", "key", key)
		os.Exit(1)
	}
	return v
}

func main() {
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, nil)))

	ctx := context.Background()

	cfg := dispatch.Config{
		ModelsURI:    env("BURN_EMULATOR_MODELS_URI"),
		InputsURI:    env("BURN_EMULATOR_INPUTS_URI"),
		OutputBucket: env("BURN_EMULATOR_OUTPUT_URI"),
		RunnerGPUJob: os.Getenv("BURN_EMULATOR_RUNNER_GPU_JOB"), // optional; unset runs DL requests on PT
		RunnerCPUJob: env("BURN_EMULATOR_RUNNER_CPU_JOB"),
	}

	client, err := dispatch.NewClient(ctx, cfg)
	if err != nil {
		slog.Error("failed to init dispatch client", "error", err)
		os.Exit(1)
	}

	jobsHandler := &handlers.JobsHandler{Dispatch: client}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/jobs", jobsHandler.Create)
	mux.HandleFunc("GET /v1/jobs/{inputs_version}/{varloc}/{model_version}/{hash}", jobsHandler.Get)
	// the Pub/Sub push subscription endpoint
	mux.Handle("POST /internal/pubsub/run-reports", &handlers.ReportHandler{Dispatch: client})
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("ok"))
	})

	srv := &http.Server{
		Addr:              ":8080",
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       15 * time.Second,
		// stays below the service request timeout
		WriteTimeout: handlers.MaxHandlerDuration + 20*time.Second,
	}

	slog.Info("burn-emulator-api listening", "addr", srv.Addr)
	if err := srv.ListenAndServe(); err != nil {
		slog.Error("server exited", "error", err)
		os.Exit(1)
	}
}
