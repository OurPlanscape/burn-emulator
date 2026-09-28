// Package dispatch runs POST /v1/jobs: resolve model + data versions, GCS
// cache + claim, then call the runner.
package dispatch

import (
	"context"
	"fmt"

	storage "google.golang.org/api/storage/v1"
)

type Config struct {
	ModelsURI    string // gs://<bucket>[/<prefix>] root of the model registry
	InputsURI    string // gs://<bucket> root of the fuels/topo inputs (reads current -> data_version)
	OutputBucket string // gs://<bucket> for outputs + the claim
	RunnerJob    string // fully-qualified burn-emulator-runner job name: projects/*/locations/*/jobs/*
}

type Client struct {
	storage       *storage.Service
	modelVersions *versionResolver
	dataVersions  *versionResolver
	runner        *runnerClient
	cfg           Config
}

func NewClient(ctx context.Context, cfg Config) (*Client, error) {
	storageSvc, err := storage.NewService(ctx)
	if err != nil {
		return nil, fmt.Errorf("creating storage client: %w", err)
	}
	modelVersions, err := newVersionResolver(storageSvc, cfg.ModelsURI)
	if err != nil {
		return nil, err
	}
	dataVersions, err := newVersionResolver(storageSvc, cfg.InputsURI)
	if err != nil {
		return nil, err
	}
	runner, err := newRunnerClient(ctx, cfg.RunnerJob)
	if err != nil {
		return nil, err
	}
	return &Client{storage: storageSvc, modelVersions: modelVersions, dataVersions: dataVersions, runner: runner, cfg: cfg}, nil
}
