package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"regexp"
	"time"

	"gopkg.in/yaml.v3"

	"burn-emulator-api/internal/auth"
	"burn-emulator-api/internal/gke"
	"burn-emulator-api/internal/rate_limit"
)

const (
	maxBodyBytes   = 1 << 12
	requestTimeout = 5 * time.Second
)

var (
	validCaching = map[string]bool{
		"none": true, "short": true, "long": true, "cdn": true,
	}
	validJobName = regexp.MustCompile(`^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$`)
)

type VariationSet map[string]bool

func (s VariationSet) Contains(name string) bool {
	return s[name]
}

type variationsConfig struct {
	Variations []string `yaml:"variations"`
}

func LoadVariations(path string) (VariationSet, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading variations file %s: %w", path, err)
	}

	var cfg variationsConfig
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parsing variations file %s: %w", path, err)
	}
	if len(cfg.Variations) == 0 {
		return nil, fmt.Errorf("variations file %s lists no variations", path)
	}

	set := make(VariationSet, len(cfg.Variations))
	for _, v := range cfg.Variations {
		set[v] = true
	}
	return set, nil
}

type jobRequestBody struct {
	Variation string `json:"variation"`
	Caching   string `json:"caching"`
	JobName   string `json:"job_name"`
}

type jobResponseBody struct {
	JobName string `json:"job_name"`
	Status  string `json:"status"`
}

type JobsHandler struct {
	K8s        *k8s.Client
	Verifier   *auth.Verifier
	Limiter    *ratelimit.Store
	Variations VariationSet
}

func (h *JobsHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	clientIP := auth.ClientIP(r)

	if !h.Verifier.Allow(r) {
		log.Printf("auth failure from %s", clientIP)
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return
	}

	if !h.Limiter.Allow(clientIP) {
		http.Error(w, "rate limit exceeded", http.StatusTooManyRequests)
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, maxBodyBytes)

	var body jobRequestBody
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}

	if err := validate(body, h.Variations); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), requestTimeout)
	defer cancel()

	jobName, err := h.K8s.CreateJob(ctx, k8s.JobRequest{
		Variation: body.Variation,
		Caching:   body.Caching,
		JobName:   body.JobName,
	})
	if err != nil {
		log.Printf("job creation failed: %v", err) // don't leak internals to the caller
		http.Error(w, "failed to schedule job", http.StatusInternalServerError)
		return
	}

	log.Printf("job created: name=%s variation=%s caching=%s by=%s",
		jobName, body.Variation, body.Caching, clientIP)

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	json.NewEncoder(w).Encode(jobResponseBody{JobName: jobName, Status: "scheduled"})
}

func validate(body jobRequestBody, allowed VariationSet) error {
	if !allowed.Contains(body.Variation) {
		return errors.New("invalid 'variation': not in the configured allow-list")
	}
	if !validCaching[body.Caching] {
		return errors.New("invalid 'caching': must be one of none|short|long|cdn")
	}
	if !validJobName.MatchString(body.JobName) {
		return errors.New("invalid 'job_name': must be 1-63 lowercase alphanumeric characters or '-', starting/ending with alphanumeric")
	}
	return nil
}