package dispatch

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"google.golang.org/api/idtoken"
)

// calls the burn-emulator-runner GPU service.
type runnerClient struct {
	http *http.Client
	url  string // base URL
}

// build a runner client whose requests carry an OIDC ID token for the
// runner's audience.
func newRunnerClient(ctx context.Context, url string) (*runnerClient, error) {
	url = strings.TrimSuffix(url, "/")
	hc, err := idtoken.NewClient(ctx, url)
	if err != nil {
		return nil, fmt.Errorf("creating runner id-token client: %w", err)
	}
	return &runnerClient{http: hc, url: url}, nil
}

// the JSON body POSTed to the runner's /infer.
type inferRequest struct {
	VarLoc           string   `json:"varloc"`
	Version          string   `json:"version"`
	TreatmentArea    string   `json:"treatment_area"`
	TreatmentAreaCRS string   `json:"treatment_area_crs"`
	IgnitionDensity  *float64 `json:"ignition_density,omitempty"`
	Hash             string   `json:"hash"`
	OutputPath       string   `json:"output_path"`
}

// how often Ready re-polls /healthz while waiting for the runner to come up.
const readyPollInterval = 3 * time.Second

// block until the runner's /healthz returns 200, or ctx is done.
func (r *runnerClient) Ready(ctx context.Context) error {
	var last string
	for {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, r.url+"/healthz", nil)
		if err != nil {
			return err
		}
		resp, err := r.http.Do(req)
		if err != nil {
			last = err.Error()
		} else {
			io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<10))
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
			last = resp.Status
		}

		select {
		case <-ctx.Done():
			return fmt.Errorf("runner not ready (last: %s): %w", last, ctx.Err())
		case <-time.After(readyPollInterval):
		}
	}
}

// POST /infer and block until the run completes.
func (r *runnerClient) Infer(ctx context.Context, req inferRequest) error {
	body, err := json.Marshal(req)
	if err != nil {
		return err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, r.url+"/infer", bytes.NewReader(body))
	if err != nil {
		return err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := r.http.Do(httpReq)
	if err != nil {
		return fmt.Errorf("calling runner: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<12))
		return fmt.Errorf("runner returned %s: %s", resp.Status, bytes.TrimSpace(msg))
	}
	return nil
}
