package dispatch

import (
	"context"
	"fmt"
	"log/slog"
	"strconv"

	run "google.golang.org/api/run/v2"
)

// must match both runner jobs' "inputs" GCS volume mount_path in Terraform.
const inputsMountPath = "/inputs"

type runnerClient struct {
	svc  *run.Service
	jobs map[string]string // backend -> fully-qualified job name: projects/*/locations/*/jobs/*
}

func newRunnerClient(ctx context.Context, gpuJob, cpuJob string) (*runnerClient, error) {
	svc, err := run.NewService(ctx)
	if err != nil {
		return nil, fmt.Errorf("creating run client: %w", err)
	}
	jobs := map[string]string{BackendPT: cpuJob}
	if gpuJob != "" {
		jobs[BackendDL] = gpuJob
	}
	return &runnerClient{svc: svc, jobs: jobs}, nil
}

// passed to the job execution as env overrides
type inferRequest struct {
	InputsVersion     string
	VarLoc            string
	ModelVersion      string // only sent for DL
	TreatmentAreaPath string // gs:// geojson next to the output, see treatmentAreaObject
	IgnitionDensity   *float64
	Backend           string
	Hash              string
	OutputPath        string
	ReportPath        string // gs:// path of the _reports/ report the runner writes when it finishes
	ClaimGeneration   int64
}

// starts a runner job execution without waiting; the runner reports via _reports/
func (r *runnerClient) Trigger(ctx context.Context, req inferRequest) error {
	job, ok := r.jobs[req.Backend]
	if !ok {
		return fmt.Errorf("no runner job for backend %q", req.Backend)
	}
	env := []*run.GoogleCloudRunV2EnvVar{
		{Name: "BURN_EMULATOR_VARLOC", Value: req.VarLoc},
		{Name: "BURN_EMULATOR_INPUTS_VERSION", Value: req.InputsVersion},
		{Name: "BURN_EMULATOR_TREATMENT_AREA_PATH", Value: req.TreatmentAreaPath},
		{Name: "BURN_EMULATOR_HASH", Value: req.Hash},
		{Name: "BURN_EMULATOR_BACKEND", Value: req.Backend},
		{Name: "BURN_EMULATOR_OUTPUT_PATH", Value: req.OutputPath},
		{Name: "BURN_EMULATOR_REPORT_PATH", Value: req.ReportPath},
		{Name: "BURN_EMULATOR_CLAIM_GENERATION", Value: strconv.FormatInt(req.ClaimGeneration, 10)},
		{Name: "BURN_EMULATOR_BASELINE_FUELS", Value: fmt.Sprintf("%s/%s/baseline", inputsMountPath, req.InputsVersion)},
		{Name: "BURN_EMULATOR_LEGALMAX_FUELS", Value: fmt.Sprintf("%s/%s/legalmax", inputsMountPath, req.InputsVersion)},
		{Name: "BURN_EMULATOR_TOPO_PATH", Value: fmt.Sprintf("%s/%s/topo", inputsMountPath, req.InputsVersion)},
		{Name: "BURN_EMULATOR_FBFM_MAP_PATH", Value: fmt.Sprintf("%s/%s/fbfm/fbfm_behavior_adjectives.csv", inputsMountPath, req.InputsVersion)},
	}
	if req.Backend == BackendDL {
		env = append(env, &run.GoogleCloudRunV2EnvVar{Name: "BURN_EMULATOR_MODEL_VERSION", Value: req.ModelVersion})
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

	op, err := r.svc.Projects.Locations.Jobs.Run(job, body).Context(ctx).Do()
	if err != nil {
		return fmt.Errorf("triggering runner job: %w", err)
	}
	slog.Info("runner job triggered", "operation", op.Name, "backend", req.Backend, "hash", req.Hash)
	return nil
}
