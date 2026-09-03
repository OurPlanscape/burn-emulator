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
	"strings"
	"time"

	"burn-emulator-api/internal/dispatch"
)

const (
	// treatment_area is inline GeoJSON which might be big
	maxBodyBytes = 1 << 20
	// a request blocks on the synchronous GPU run (cold model load + inference,
	// plus a possible runner cold start under scale-out).
	requestTimeout = 120 * time.Second
)

var validJobName = regexp.MustCompile(`^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$`)

// the allow-list of accepted varloc values.
type VarLocSet map[string]bool

// report whether name is in the allow-list.
func (s VarLocSet) Contains(name string) bool {
	return s[name]
}

// read and parse the varlocs.txt allow-list at path: one varloc per line.
func LoadVarLocs(path string) (VarLocSet, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading varlocs file %s: %w", path, err)
	}

	set := make(VarLocSet)
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		set[line] = true
	}
	if len(set) == 0 {
		return nil, fmt.Errorf("varlocs file %s lists no varlocs", path)
	}
	return set, nil
}

// the JSON body accepted by POST /v1/jobs.
type jobRequestBody struct {
	TreatmentArea string `json:"treatment_area"`
	VarLoc        string `json:"varloc"`
	JobName       string `json:"job_name"`
}

// the JSON body returned by POST /v1/jobs.
type jobResponseBody struct {
	JobName      string `json:"job_name,omitempty"`
	Hash         string `json:"hash"`
	ModelVersion string `json:"model_version"`
	Status       string `json:"status"`
	VarLoc       string `json:"varloc"`
	Cached       bool   `json:"cached"`
	Attempts     int    `json:"attempts,omitempty"`
	OutputPath   string `json:"output_path"`
}

// serve POST /v1/jobs. Caller identity is verified upstream, not here.
type JobsHandler struct {
	Dispatch *dispatch.Client
	VarLocs  VarLocSet
}

// validate the request, run it, and return the status, parameter hash, model
// version, and output path.
func (h *JobsHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	clientAddr := clientIP(r)

	r.Body = http.MaxBytesReader(w, r.Body, maxBodyBytes)

	var body jobRequestBody
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			http.Error(w, fmt.Sprintf("request body exceeds %d bytes", maxBodyBytes), http.StatusRequestEntityTooLarge)
			return
		}
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}

	if err := validate(body, h.VarLocs); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), requestTimeout)
	defer cancel()

	result, err := h.Dispatch.CreateJob(ctx, dispatch.JobRequest{
		TreatmentArea: body.TreatmentArea,
		VarLoc:        body.VarLoc,
		JobName:       body.JobName,
	})
	if err != nil {
		if errors.Is(err, context.Canceled) && r.Context().Err() != nil {
			// client hung up (or the server is shutting down); the runner keeps
			// going and the result lands in the cache for the retry.
			slog.Info("request cancelled, run continues server-side",
				"job_name", body.JobName, "varloc", body.VarLoc, "client_ip", clientAddr)
			return
		}
		slog.Error("run failed", "error", err, "job_name", body.JobName, "varloc", body.VarLoc, "client_ip", clientAddr)
		http.Error(w, "failed to run burn emulation", http.StatusInternalServerError)
		return
	}

	statusCode := http.StatusOK
	if result.Status == "pending" {
		statusCode = http.StatusAccepted
	}

	slog.Info("run handled",
		"job_name", result.JobName, "varloc", body.VarLoc, "hash", result.Hash,
		"model_version", result.ModelVersion, "status", result.Status, "attempts", result.Attempts,
		"output_path", result.OutputPath, "client_ip", clientAddr)

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(jobResponseBody{
		JobName:      result.JobName,
		Hash:         result.Hash,
		ModelVersion: result.ModelVersion,
		Status:       result.Status,
		VarLoc:       body.VarLoc,
		Cached:       result.Status == "cached",
		Attempts:     result.Attempts,
		OutputPath:   result.OutputPath,
	})
}

// check the varloc against the allow-list and job_name against the label rules
// (job_name is stored in the claim ledger and echoed back).
func validate(body jobRequestBody, allowed VarLocSet) error {
	if !allowed.Contains(body.VarLoc) {
		return errors.New("invalid 'varloc': not in the configured allow-list")
	}
	if !validJobName.MatchString(body.JobName) {
		return errors.New("invalid 'job_name': must be 1-63 lowercase alphanumeric characters or '-', starting/ending with alphanumeric")
	}
	if strings.TrimSpace(body.TreatmentArea) == "" {
		return errors.New("missing 'treatment_area'")
	}
	return nil
}

// extract the caller's address for logging. X-Forwarded-For is trusted
// because only internal callers reach this API.
func clientIP(r *http.Request) string {
	if ip := r.Header.Get("X-Forwarded-For"); ip != "" {
		return ip
	}
	return r.RemoteAddr
}
