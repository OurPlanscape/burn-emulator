package handlers

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"strconv"
	"time"

	"burn-emulator-api/internal/dispatch"
)

// HandleReport's detached claim calls (ownsClaim + deleteOutput, or releaseRun)
// can add up to 2x dispatch's releaseTimeout (10s) on top, so the worst case is
// ~110s; keep the Pub/Sub subscription's ack_deadline_seconds (120s) above that.
const reportTimeout = 90 * time.Second

type pushEnvelope struct {
	Message struct {
		MessageID  string            `json:"messageId"`
		Attributes map[string]string `json:"attributes"`
	} `json:"message"`
}

// serve POST /internal/pubsub/run-reports
// NOTE: Any run.invoker can reach this route but should only do so for legit reports
type ReportHandler struct {
	Dispatch *dispatch.Client
}

func (h *ReportHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, maxBodyBytes)

	// malformed or irrelevant deliveries are acked so they are not redelivered.
	var env pushEnvelope
	if err := json.NewDecoder(r.Body).Decode(&env); err != nil {
		slog.Warn("dropping malformed push delivery", "error", err)
		w.WriteHeader(http.StatusNoContent)
		return
	}
	attrs := env.Message.Attributes
	if attrs["eventType"] != "OBJECT_FINALIZE" {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	generation, err := strconv.ParseInt(attrs["objectGeneration"], 10, 64)
	if err != nil {
		slog.Warn("dropping push delivery without object generation", "message_id", env.Message.MessageID)
		w.WriteHeader(http.StatusNoContent)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), reportTimeout)
	defer cancel()

	if err := h.Dispatch.HandleReport(ctx, attrs["bucketId"], attrs["objectId"], generation); err != nil {
		slog.Error("failed to process run notification", "error", err,
			"message_id", env.Message.MessageID, "object", attrs["objectId"])
		http.Error(w, "failed to process notification", http.StatusInternalServerError)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}
