package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"regexp"
	"strings"
	"time"

	"burn-emulator-api/internal/dispatch"
)

const (
	maxBodyBytes = 1 << 20
	// caller-attached steps; the claim + trigger run under dispatch.DetachedBudget
	requestTimeout = 2 * time.Minute

	// http.Server.WriteTimeout builds on this
	MaxHandlerDuration = requestTimeout + dispatch.DetachedBudget
)

var validJobName = regexp.MustCompile(`^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$`)

var validVarLoc = regexp.MustCompile(`^[A-Za-z0-9]{1,32}$`)

type jobRequestBody struct {
	TreatmentArea   string   `json:"treatment_area"`
	VarLoc          string   `json:"varloc,omitempty"`
	JobName         string   `json:"job_name,omitempty"`
	IgnitionDensity *float64 `json:"ignition_density,omitempty"`
	Backend         string   `json:"backend,omitempty"`
}

type jobResponseBody struct {
	JobID         string `json:"job_id"`
	JobName       string `json:"job_name,omitempty"`
	Hash          string `json:"hash"`
	InputsVersion string `json:"inputs_version"`
	VarLoc        string `json:"varloc"`
	ModelVersion  string `json:"model_version,omitempty"`
	Backend       string `json:"backend"`
	Status        string `json:"status"`
	Cached        bool   `json:"cached"`
	Attempts      int    `json:"attempts,omitempty"`
	OutputPath    string `json:"output_path"`
	Error         string `json:"error,omitempty"`
}

// caller identity is verified by Cloud Run IAM
type JobsHandler struct {
	Dispatch *dispatch.Client
}

// POST /v1/jobs
func (h *JobsHandler) Create(w http.ResponseWriter, r *http.Request) {
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

	if body.Backend == "" {
		body.Backend = dispatch.BackendDL
	}
	if err := validate(body); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	treatmentArea, err := normalizeGeoJSON(body.TreatmentArea)
	if err != nil {
		http.Error(w, "invalid 'treatment_area': must be a GeoJSON object", http.StatusBadRequest)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), requestTimeout)
	defer cancel()

	result, err := h.Dispatch.CreateJob(ctx, dispatch.JobRequest{
		TreatmentArea:   treatmentArea,
		VarLoc:          strings.ToUpper(body.VarLoc),
		JobName:         body.JobName,
		IgnitionDensity: body.IgnitionDensity,
		Backend:         body.Backend,
	})
	if errors.Is(err, dispatch.ErrInvalidTreatmentArea) || errors.Is(err, dispatch.ErrOutsideVarLocs) ||
		errors.Is(err, dispatch.ErrUnknownVarLoc) || errors.Is(err, dispatch.ErrOutsideVarLoc) {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if err != nil {
		if errors.Is(err, context.Canceled) && r.Context().Err() != nil {
			slog.Info("request cancelled",
				"job_name", body.JobName, "client_ip", clientAddr)
			return
		}
		slog.Error("run failed", "error", err, "job_name", body.JobName, "client_ip", clientAddr)
		http.Error(w, "failed to run burn emulation", http.StatusInternalServerError)
		return
	}

	slog.Info("run handled",
		"job_id", result.ID.Path(), "job_name", result.JobName, "status", result.Status,
		"attempts", result.Attempts, "output_path", result.OutputPath, "client_ip", clientAddr)

	statusCode := http.StatusOK
	if result.Status == "pending" {
		statusCode = http.StatusAccepted
		w.Header().Set("Location", "/v1/jobs/"+result.ID.Path())
	}
	writeJob(w, statusCode, result)
}

// GET /v1/jobs/{inputs_version}/{varloc}/{model_version}/{hash}
func (h *JobsHandler) Get(w http.ResponseWriter, r *http.Request) {
	id := dispatch.JobID{
		InputsVersion: r.PathValue("inputs_version"),
		VarLoc:        r.PathValue("varloc"),
		ModelVersion:  r.PathValue("model_version"),
		Hash:          r.PathValue("hash"),
	}
	ctx, cancel := context.WithTimeout(r.Context(), requestTimeout)
	defer cancel()

	result, err := h.Dispatch.GetJob(ctx, id)
	if errors.Is(err, dispatch.ErrJobNotFound) {
		http.Error(w, "job not found", http.StatusNotFound)
		return
	}
	if err != nil {
		slog.Error("job lookup failed", "error", err, "job_id", id.Path(), "client_ip", clientIP(r))
		http.Error(w, "failed to look up job", http.StatusInternalServerError)
		return
	}
	writeJob(w, http.StatusOK, result)
}

func writeJob(w http.ResponseWriter, statusCode int, result dispatch.JobResult) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	modelVersion := result.ID.ModelVersion
	if result.ID.Backend() == dispatch.BackendPT {
		modelVersion = ""
	}
	json.NewEncoder(w).Encode(jobResponseBody{
		JobID:         result.ID.Path(),
		JobName:       result.JobName,
		Hash:          result.ID.Hash,
		InputsVersion: result.ID.InputsVersion,
		VarLoc:        result.ID.VarLoc,
		ModelVersion:  modelVersion,
		Backend:       result.ID.Backend(),
		Status:        result.Status,
		Cached:        result.Status == "cached",
		Attempts:      result.Attempts,
		OutputPath:    result.OutputPath,
		Error:         result.Error,
	})
}

// job_name is stored as claim metadata
func validate(body jobRequestBody) error {
	if body.VarLoc != "" && !validVarLoc.MatchString(body.VarLoc) {
		return errors.New("invalid 'varloc': must be 1-32 alphanumeric characters")
	}
	if body.JobName != "" && !validJobName.MatchString(body.JobName) {
		return errors.New("invalid 'job_name': must be 1-63 lowercase alphanumeric characters or '-', starting/ending with alphanumeric")
	}
	if strings.TrimSpace(body.TreatmentArea) == "" {
		return errors.New(
			"missing 'treatment_area'",
		)
	}
	if body.IgnitionDensity != nil && *body.IgnitionDensity <= 0 {
		return errors.New("invalid 'ignition_density': must be > 0")
	}
	if body.Backend != dispatch.BackendDL && body.Backend != dispatch.BackendPT {
		return fmt.Errorf("invalid 'backend': must be %q or %q", dispatch.BackendDL, dispatch.BackendPT)
	}
	return nil
}

// sorted keys, no whitespace
func normalizeGeoJSON(s string) (string, error) {
	var v any
	if err := json.Unmarshal([]byte(s), &v); err != nil {
		return "", err
	}
	if _, ok := v.(map[string]any); !ok {
		return "", errors.New("not a JSON object")
	}
	b, err := json.Marshal(v)
	return string(b), err
}

func clientIP(r *http.Request) string {
	if ip := r.Header.Get("X-Forwarded-For"); ip != "" {
		return ip
	}
	return r.RemoteAddr
}
