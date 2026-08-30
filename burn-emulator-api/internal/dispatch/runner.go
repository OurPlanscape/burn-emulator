package dispatch

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

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
	VarLoc        string `json:"varloc"`
	Version       string `json:"version"`
	TreatmentArea string `json:"treatment_area"`
	Hash          string `json:"hash"`
	OutputPath    string `json:"output_path"`
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
