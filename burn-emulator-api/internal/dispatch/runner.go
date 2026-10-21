package dispatch

import (
	"context"
	"fmt"
	"log/slog"
	"maps"
	"slices"
	"strconv"

	run "google.golang.org/api/run/v2"
)

const (
	inputsMountPath = "/inputs" // runner's inputs bucket mount
	modelsMountPath = "/models" // runner's models bucket mount
)

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

// passed to the job execution as env overrides, or stored as the claim body in metal mode
type inferRequest struct {
	InputsVersion   string
	VarLoc          string
	ModelVersion    string // the bundle; only sent when there is one
	BundleURI       string // gs:// of that bundle; recorded in the run's meta.geojson
	OutputMeta      string // JobResult.OutputMeta
	IgnitionDensity *float64
	Backend         string
	Hash            string
	OutputPath      string // JobResult.OutputPath
	ReportPath      string // gs:// path of the _reports/ report the runner writes when it finishes
	ClaimGeneration int64
	Metal           bool // the runner loop sets the inputs and bundle paths from its on-prem roots
}

// the runner's env; BURN_EMULATOR_CLAIM_GENERATION only when set (the metal loop uses the claim's generation)
func runnerEnv(req inferRequest) map[string]string {
	env := map[string]string{
		"BURN_EMULATOR_VARLOC":         req.VarLoc,
		"BURN_EMULATOR_INPUTS_VERSION": req.InputsVersion,
		"BURN_EMULATOR_OUTPUT_META":    req.OutputMeta,
		"BURN_EMULATOR_HASH":           req.Hash,
		"BURN_EMULATOR_BACKEND":        req.Backend,
		"BURN_EMULATOR_OUTPUT_PATH":    req.OutputPath,
		"BURN_EMULATOR_REPORT_PATH":    req.ReportPath,
	}
	if !req.Metal {
		env["BURN_EMULATOR_BASELINE_FUELS"] = fmt.Sprintf("%s/%s/baseline", inputsMountPath, req.InputsVersion)
		env["BURN_EMULATOR_LEGALMAX_FUELS"] = fmt.Sprintf("%s/%s/legalmax", inputsMountPath, req.InputsVersion)
		env["BURN_EMULATOR_TOPO_PATH"] = fmt.Sprintf("%s/%s/topo", inputsMountPath, req.InputsVersion)
		env["BURN_EMULATOR_FBFM_MAP_PATH"] = fmt.Sprintf("%s/%s/fbfm/fbfm_behavior_adjectives.csv", inputsMountPath, req.InputsVersion)
	}
	if req.ClaimGeneration != 0 {
		env["BURN_EMULATOR_CLAIM_GENERATION"] = strconv.FormatInt(req.ClaimGeneration, 10)
	}
	if req.ModelVersion != "" {
		env["BURN_EMULATOR_MODEL_VERSION"] = req.ModelVersion
		env["BURN_EMULATOR_BUNDLE_URI"] = req.BundleURI
		if !req.Metal {
			env["BURN_EMULATOR_BUNDLE_DIR"] = fmt.Sprintf("%s/%s/%s", modelsMountPath, req.VarLoc, req.ModelVersion)
		}
	}
	if req.IgnitionDensity != nil {
		env["BURN_EMULATOR_IGNITION_DENSITY"] = strconv.FormatFloat(*req.IgnitionDensity, 'g', -1, 64)
	}
	return env
}

// starts a runner job execution without waiting; the runner reports via _reports/
func (r *runnerClient) Trigger(ctx context.Context, req inferRequest) error {
	job, ok := r.jobs[req.Backend]
	if !ok {
		return fmt.Errorf("no runner job for backend %q", req.Backend)
	}
	vars := runnerEnv(req)
	names := slices.Sorted(maps.Keys(vars))
	env := make([]*run.GoogleCloudRunV2EnvVar, 0, len(names))
	for _, name := range names {
		env = append(env, &run.GoogleCloudRunV2EnvVar{Name: name, Value: vars[name]})
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
