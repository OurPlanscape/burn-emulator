package dispatch

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strconv"
	"time"

	run "google.golang.org/api/run/v2"
)

type runnerClient struct {
	svc *run.Service
	job string // fully-qualified job name: projects/*/locations/*/jobs/*
}

func newRunnerClient(ctx context.Context, job string) (*runnerClient, error) {
	svc, err := run.NewService(ctx)
	if err != nil {
		return nil, fmt.Errorf("creating run client: %w", err)
	}
	return &runnerClient{svc: svc, job: job}, nil
}

// a single inference request, passed to the job execution as env var overrides.
type inferRequest struct {
	VarLoc           string
	Version          string
	FuelsVersion     string
	TopoVersion      string
	TreatmentArea    string
	TreatmentAreaCRS string
	IgnitionDensity  *float64
	Hash             string
	OutputPath       string
}

// must match the runner job's "inputs" GCS volume mount_path in Terraform.
const inputsMountPath = "/inputs"

const runPollInterval = 3 * time.Second

// how long a best-effort execution cancel gets once the caller's ctx is done.
const cancelTimeout = 10 * time.Second

func (r *runnerClient) Run(ctx context.Context, req inferRequest) error {
	env := []*run.GoogleCloudRunV2EnvVar{
		{Name: "BURN_EMULATOR_VARLOC", Value: req.VarLoc},
		{Name: "BURN_EMULATOR_VERSION", Value: req.Version},
		{Name: "BURN_EMULATOR_TREATMENT_AREA", Value: req.TreatmentArea},
		{Name: "BURN_EMULATOR_TREATMENT_AREA_CRS", Value: req.TreatmentAreaCRS},
		{Name: "BURN_EMULATOR_HASH", Value: req.Hash},
		{Name: "BURN_EMULATOR_OUTPUT_PATH", Value: req.OutputPath},
		{Name: "BURN_EMULATOR_BASELINE_FUELS", Value: fmt.Sprintf("%s/%s/baseline", inputsMountPath, req.FuelsVersion)},
		{Name: "BURN_EMULATOR_LEGALMAX_FUELS", Value: fmt.Sprintf("%s/%s/legalmax", inputsMountPath, req.FuelsVersion)},
		{Name: "BURN_EMULATOR_TOPO_PATH", Value: fmt.Sprintf("%s/%s/topo", inputsMountPath, req.TopoVersion)},
	}
	if req.IgnitionDensity != nil {
		env = append(env, &run.GoogleCloudRunV2EnvVar{
			Name:  "BURN_EMULATOR_IGNITION_DENSITY",
			Value: strconv.FormatFloat(*req.IgnitionDensity, 'g', -1, 64),
		})
	}

	body := &run.GoogleCloudRunV2RunJobRequest{
		Overrides: &run.GoogleCloudRunV2Overrides{
			TaskCount: 1,
			ContainerOverrides: []*run.GoogleCloudRunV2ContainerOverride{
				{Name: "runner", Env: env},
			},
		},
	}

	op, err := r.svc.Projects.Locations.Jobs.Run(r.job, body).Context(ctx).Do()
	if err != nil {
		return fmt.Errorf("triggering runner job: %w", err)
	}

	for !op.Done {
		select {
		case <-ctx.Done():
			r.cancelExecution(op)
			return fmt.Errorf("waiting for runner job execution: %w", ctx.Err())
		case <-time.After(runPollInterval):
		}
		op, err = r.svc.Projects.Locations.Operations.Get(op.Name).Context(ctx).Do()
		if err != nil {
			return fmt.Errorf("polling runner job execution: %w", err)
		}
	}

	if op.Error != nil {
		return fmt.Errorf("runner job execution failed: %s", op.Error.Message)
	}
	return nil
}

// best-effort cancel of the execution behind a still-running operation, once
// the caller has given up waiting on it (e.g. the HTTP client disconnected).
func (r *runnerClient) cancelExecution(op *run.GoogleLongrunningOperation) {
	var exec run.GoogleCloudRunV2Execution
	if err := json.Unmarshal(op.Metadata, &exec); err != nil || exec.Name == "" {
		slog.Warn("could not determine runner execution to cancel", "operation", op.Name, "error", err)
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), cancelTimeout)
	defer cancel()
	req := &run.GoogleCloudRunV2CancelExecutionRequest{}
	if _, err := r.svc.Projects.Locations.Jobs.Executions.Cancel(exec.Name, req).Context(ctx).Do(); err != nil {
		slog.Warn("failed to cancel runner job execution", "execution", exec.Name, "error", err)
	}
}
