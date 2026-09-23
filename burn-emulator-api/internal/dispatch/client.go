// Package dispatch runs POST /v1/jobs: resolve version, GCS cache + claim
// ledger, then call the runner.
package dispatch

import (
	"context"
	"fmt"

	storage "google.golang.org/api/storage/v1"
)

type Config struct {
	ModelsURI    string // gs://<bucket>[/<prefix>] root of the model registry
	InputsURI    string // gs://<bucket> root of the fuels/topo inputs (reads fuels/current, topo/current)
	OutputBucket string // gs://<bucket> for outputs + the claim ledger
	RunnerJob    string // fully-qualified burn-emulator-runner job name: projects/*/locations/*/jobs/*
}

type Client struct {
	storage       *storage.Service
	versions      *versionResolver
	inputVersions *versionResolver
	runner        *runnerClient
	cfg           Config
}

func NewClient(ctx context.Context, cfg Config) (*Client, error) {
	storageSvc, err := storage.NewService(ctx)
	if err != nil {
		return nil, fmt.Errorf("creating storage client: %w", err)
	}
	versions, err := newVersionResolver(storageSvc, cfg.ModelsURI)
	if err != nil {
		return nil, err
	}
	inputVersions, err := newVersionResolver(storageSvc, cfg.InputsURI)
	if err != nil {
		return nil, err
	}
	runner, err := newRunnerClient(ctx, cfg.RunnerJob)
	if err != nil {
		return nil, err
	}
	return &Client{storage: storageSvc, versions: versions, inputVersions: inputVersions, runner: runner, cfg: cfg}, nil
}
