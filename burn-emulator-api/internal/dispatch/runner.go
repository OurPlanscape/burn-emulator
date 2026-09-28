package dispatch

import (
	"context"
	"fmt"
	"log/slog"
	"strconv"

	run "google.golang.org/api/run/v2"
)

// must match the runner job's "inputs" GCS volume mount_path in Terraform.
const inputsMountPath = "/inputs"

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
	ModelVersion     string
	DataVersion      string
	TreatmentArea    string
	TreatmentAreaCRS string
	IgnitionDensity  *float64
	Hash             string
	OutputPath       string
	ReportPath       string // gs:// path of the _reports/ report the runner writes when it finishes
	ClaimGeneration  int64
}

// start a runner job execution without waiting on it; the runner reports
// back through its _reports/ report (see report.go).
func (r *runnerClient) Trigger(ctx context.Context, req inferRequest) error {
	env := []*run.GoogleCloudRunV2EnvVar{
		{Name: "BURN_EMULATOR_VARLOC", Value: req.VarLoc},
		{Name: "BURN_EMULATOR_MODEL_VERSION", Value: req.ModelVersion},
		{Name: "BURN_EMULATOR_TREATMENT_AREA", Value: req.TreatmentArea},
		{Name: "BURN_EMULATOR_TREATMENT_AREA_CRS", Value: req.TreatmentAreaCRS},
		{Name: "BURN_EMULATOR_HASH", Value: req.Hash},
		{Name: "BURN_EMULATOR_OUTPUT_PATH", Value: req.OutputPath},
		{Name: "BURN_EMULATOR_REPORT_PATH", Value: req.ReportPath},
		{Name: "BURN_EMULATOR_CLAIM_GENERATION", Value: strconv.FormatInt(req.ClaimGeneration, 10)},
		{Name: "BURN_EMULATOR_BASELINE_FUELS", Value: fmt.Sprintf("%s/%s/baseline", inputsMountPath, req.DataVersion)},
		{Name: "BURN_EMULATOR_LEGALMAX_FUELS", Value: fmt.Sprintf("%s/%s/legalmax", inputsMountPath, req.DataVersion)},
		{Name: "BURN_EMULATOR_TOPO_PATH", Value: fmt.Sprintf("%s/%s/topo", inputsMountPath, req.DataVersion)},
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
	slog.Info("runner job triggered", "operation", op.Name, "hash", req.Hash)
	return nil
}
