package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
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
	TreatmentArea 		string	`json:"treatment_area"`
	TreatmentBuff 		float64	`json:"treatment_buff"`
	TreatmentSeed 		float64	`json:"treatment_seed"`
	IgnitionDensity 	float64	`json:"ignition_density"`
	Variation 				string	`json:"variation"`
	JobName   				string	`json:"job_name"`
}

type jobResponseBody struct {
	JobName    string `json:"job_name,omitempty"`
	Status     string `json:"status"`
	Variation  string `json:"variation"`
	Cached     bool   `json:"cached"`
	Attempts   int    `json:"attempts,omitempty"`
	OutputPath string `json:"output_path"`
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
		slog.Warn("auth failure", "client_ip", clientIP)
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

	result, err := h.K8s.CreateJob(ctx, k8s.JobRequest{
		TreatmentArea:   body.TreatmentArea,
		TreatmentBuff:   float32(body.TreatmentBuff),
		TreatmentSeed:   float32(body.TreatmentSeed),
		IgnitionDensity: float32(body.IgnitionDensity),
		Variation:       body.Variation,
		JobName:         body.JobName,
	})
	if err != nil {
		slog.Error("job creation failed", "error", err, "job_name", body.JobName, "client_ip", clientIP)
		http.Error(w, "failed to schedule job", http.StatusInternalServerError)
		return
	}

	statusCode := http.StatusAccepted
	if result.Status == "cached" {
		statusCode = http.StatusOK
	}

	slog.Info("job request handled",
		"job_name", result.JobName, "variation", body.Variation,
		"status", result.Status, "attempts", result.Attempts, "output_path", result.OutputPath, "client_ip", clientIP)

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(jobResponseBody{
		JobName:    result.JobName,
		Status:     result.Status,
		Variation:  body.Variation,
		Cached:     result.Status == "cached",
		Attempts:   result.Attempts,
		OutputPath: result.OutputPath,
	})
}

func validate(body jobRequestBody, allowed VariationSet) error {
	if !allowed.Contains(body.Variation) {
		return errors.New("invalid 'variation': not in the configured allow-list")
	}
	if !validJobName.MatchString(body.JobName) {
		return errors.New("invalid 'job_name': must be 1-63 lowercase alphanumeric characters or '-', starting/ending with alphanumeric")
	}
	return nil
}
