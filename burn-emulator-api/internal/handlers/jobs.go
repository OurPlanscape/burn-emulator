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
	// treatment_area is inline GeoJSON which might be big
	maxBodyBytes = 1 << 20
	// covers the caller-attached steps: version resolution and the cache check
	// (POST), or the output + claim lookup (GET). The claim + trigger run
	// detached, under dispatch.DetachedBudget.
	// if this takes longer than two minutes, something is really wrong and the request should be cancelled.
	requestTimeout = 2 * time.Minute

	// worst case for one request: the attached steps, then dispatch's detached
	// budget. http.Server.WriteTimeout is built on top of this.
	MaxHandlerDuration = requestTimeout + dispatch.DetachedBudget
)

var validJobName = regexp.MustCompile(`^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$`)

type jobRequestBody struct {
	TreatmentArea    string   `json:"treatment_area"`
	TreatmentAreaCRS string   `json:"treatment_area_crs"`
	VarLoc           string   `json:"varloc"`
	JobName          string   `json:"job_name"`
	IgnitionDensity  *float64 `json:"ignition_density,omitempty"`
}

type jobResponseBody struct {
	JobID        string `json:"job_id"`
	JobName      string `json:"job_name,omitempty"`
	Hash         string `json:"hash"`
	ModelVersion string `json:"model_version"`
	DataVersion  string `json:"data_version"`
	Status       string `json:"status"`
	VarLoc       string `json:"varloc"`
	Cached       bool   `json:"cached"`
	Attempts     int    `json:"attempts,omitempty"`
	OutputPath   string `json:"output_path"`
	Error        string `json:"error,omitempty"`
}

// serve /v1/jobs. Caller identity is verified upstream, not here.
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

	if err := validate(body); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), requestTimeout)
	defer cancel()

	result, err := h.Dispatch.CreateJob(ctx, dispatch.JobRequest{
		TreatmentArea:    body.TreatmentArea,
		TreatmentAreaCRS: body.TreatmentAreaCRS,
		VarLoc:           body.VarLoc,
		JobName:          body.JobName,
		IgnitionDensity:  body.IgnitionDensity,
	})
	if errors.Is(err, dispatch.ErrUnknownVarLoc) {
		http.Error(w, "invalid 'varloc': not in the published allow-list", http.StatusBadRequest)
		return
	}
	if err != nil {
		if errors.Is(err, context.Canceled) && r.Context().Err() != nil {
			// client hung up before the run was claimed.
			slog.Info("request cancelled",
				"job_name", body.JobName, "varloc", body.VarLoc, "client_ip", clientAddr)
			return
		}
		slog.Error("run failed", "error", err, "job_name", body.JobName, "varloc", body.VarLoc, "client_ip", clientAddr)
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

// GET /v1/jobs/{varloc}/{model_version}/{data_version}/{hash}
func (h *JobsHandler) Get(w http.ResponseWriter, r *http.Request) {
	id := dispatch.JobID{
		VarLoc:       r.PathValue("varloc"),
		ModelVersion: r.PathValue("model_version"),
		DataVersion:  r.PathValue("data_version"),
		Hash:         r.PathValue("hash"),
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
	json.NewEncoder(w).Encode(jobResponseBody{
		JobID:        result.ID.Path(),
		JobName:      result.JobName,
		Hash:         result.ID.Hash,
		ModelVersion: result.ID.ModelVersion,
		DataVersion:  result.ID.DataVersion,
		Status:       result.Status,
		VarLoc:       result.ID.VarLoc,
		Cached:       result.Status == "cached",
		Attempts:     result.Attempts,
		OutputPath:   result.OutputPath,
		Error:        result.Error,
	})
}

// job_name is stored in the claim, so it must be label-safe.
func validate(body jobRequestBody) error {
	if body.VarLoc == "" {
		return errors.New("missing 'varloc'")
	}
	if !validJobName.MatchString(body.JobName) {
		return errors.New("invalid 'job_name': must be 1-63 lowercase alphanumeric characters or '-', starting/ending with alphanumeric")
	}
	if strings.TrimSpace(body.TreatmentArea) == "" {
		return errors.New(
			"missing 'treatment_area'",
		)
	}
	if strings.TrimSpace(body.TreatmentAreaCRS) == "" {
		return errors.New(
			"missing 'treatment_area_crs'",
		)
	}
	if body.IgnitionDensity != nil && *body.IgnitionDensity <= 0 {
		return errors.New("invalid 'ignition_density': must be > 0")
	}
	return nil
}

// extract the caller's address for logging.
func clientIP(r *http.Request) string {
	if ip := r.Header.Get("X-Forwarded-For"); ip != "" {
		return ip
	}
	return r.RemoteAddr
}
