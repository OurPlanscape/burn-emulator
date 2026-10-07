// Package dispatch serves /v1/jobs: varloc selection, GCS cache and claim, runner trigger.
package dispatch

import (
	"context"
	"fmt"

	storage "google.golang.org/api/storage/v1"
)

type Config struct {
	ModelsURI    string // gs://<bucket>[/<prefix>] root of the model registry (reads <varloc>/current -> model_version)
	InputsURI    string // gs://<bucket> root of the fuels/topo/varlocs inputs (reads current -> inputs_version)
	OutputBucket string // gs://<bucket> for outputs + the claim
	RunnerGPUJob string // fully-qualified GPU burn-emulator-runner job name (DL): projects/*/locations/*/jobs/*
	RunnerCPUJob string // fully-qualified CPU-only burn-emulator-runner job name (PT)
}

type Client struct {
	storage        *storage.Service
	inputsVersions *versionResolver
	modelVersions  *versionResolver
	varLocs        *varLocsResolver
	runner         *runnerClient
	cfg            Config
}

func NewClient(ctx context.Context, cfg Config) (*Client, error) {
	storageSvc, err := storage.NewService(ctx)
	if err != nil {
		return nil, fmt.Errorf("creating storage client: %w", err)
	}
	inputsVersions, err := newVersionResolver(storageSvc, cfg.InputsURI)
	if err != nil {
		return nil, err
	}
	modelVersions, err := newVersionResolver(storageSvc, cfg.ModelsURI)
	if err != nil {
		return nil, err
	}
	varLocs, err := newVarLocsResolver(storageSvc, cfg.InputsURI)
	if err != nil {
		return nil, err
	}
	runner, err := newRunnerClient(ctx, cfg.RunnerGPUJob, cfg.RunnerCPUJob)
	if err != nil {
		return nil, err
	}
	return &Client{storage: storageSvc, inputsVersions: inputsVersions, modelVersions: modelVersions, varLocs: varLocs, runner: runner, cfg: cfg}, nil
}
