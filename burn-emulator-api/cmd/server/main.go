package main

import (
	"context"
	"log"
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
		log.Fatalf("missing required env var: %s", key)
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
	cfg := k8s.Config{
		Namespace:      env("BURN_EMULATOR_JOB_NAMESPACE"),
		ServiceAccount: env("BURN_EMULATOR_JOB_SERVICE_ACCOUNT"),
		ImageStore:			env("BURN_EMULATOR_ARTIFACT_STORE"),
		OutputBucket:   env("BURN_EMULATOR_OUTPUT_BUCKET"),
	}

	k8sClient, err := k8s.NewClient(cfg)
	if err != nil {
		log.Fatalf("failed to init k8s client: %v", err)
	}

	variationsPath := envDefault("VARIATIONS_FILE", "configs/variations.yaml")
	validVariations, err := handlers.LoadVariations(variationsPath)
	if err != nil {
		log.Fatalf("failed to load variations allow-list: %v", err)
	}

	ctx := context.Background()
	allowedCallers := strings.Split(env("BURN_EMULATOR_ALLOWED_CALLERS"), ",")
	verifier, err := auth.NewVerifier(ctx, env("BURN_EMULATOR_TOKEN_AUDIENCE"), allowedCallers)
	if err != nil {
		log.Fatalf("failed to init auth verifier: %v", err)
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
		ReadHeaderTimeout: 5 * time.Second, // SECURITY: mitigates slowloris-style attacks
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      10 * time.Second,
	}

	log.Println("burn-emulator-api listening on :8080")
	log.Fatal(srv.ListenAndServe())
}