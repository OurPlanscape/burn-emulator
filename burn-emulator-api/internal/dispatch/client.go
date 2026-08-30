// Package dispatch runs POST /v1/jobs: resolve version, GCS cache + claim
// ledger, then call the runner.
package dispatch

import (
	"context"
	"fmt"

	storage "google.golang.org/api/storage/v1"
)

// runtime config, set from env vars in main.go.
type Config struct {
	ModelsURI    string // gs://<bucket>[/<prefix>] root of the model registry
	OutputBucket string // gs://<bucket> for outputs + the claim ledger
	RunnerURL    string // base URL of the burn-emulator-runner service
}

// runs burn emulations via the GPU runner and tracks their cache state in GCS.
type Client struct {
	storage  *storage.Service
	versions *versionResolver
	runner   *runnerClient
	cfg      Config
}

// build the GCS and runner clients.
func NewClient(ctx context.Context, cfg Config) (*Client, error) {
	storageSvc, err := storage.NewService(ctx)
	if err != nil {
		return nil, fmt.Errorf("creating storage client: %w", err)
	}
	versions, err := newVersionResolver(storageSvc, cfg.ModelsURI)
	if err != nil {
		return nil, err
	}
	runner, err := newRunnerClient(ctx, cfg.RunnerURL)
	if err != nil {
		return nil, err
	}
	return &Client{storage: storageSvc, versions: versions, runner: runner, cfg: cfg}, nil
}
