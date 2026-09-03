package dispatch

import (
	"context"
	"fmt"
	"strings"
	"time"
)

const (
	warmupBudget = 5 * time.Minute // how long to wait for the runner's /healthz before giving up
	inferBudget  = 4 * time.Minute // how long an /infer call gets once the runner is up
)

// a validated run request from the /v1/jobs body.
type JobRequest struct {
	TreatmentArea string
	VarLoc        string
	JobName       string
}

// the outcome of CreateJob, echoed back to the caller.
type CreateJobResult struct {
	JobName      string
	Hash         string // cache key for the request parameters
	ModelVersion string // model version the run resolved to
	Status       string // "cached" | "pending" | "completed"
	Attempts     int
	OutputPath   string
}

// resolve the model version, check the output cache, claim the run, then run
// the burn emulation synchronously on the GPU runner.
func (c *Client) CreateJob(ctx context.Context, req JobRequest) (CreateJobResult, error) {
	version, err := c.versions.resolve(ctx, req.VarLoc)
	if err != nil {
		return CreateJobResult{}, fmt.Errorf("resolving model version for %s: %w", req.VarLoc, err)
	}

	key := CacheKey(req)
	bucket := strings.TrimSuffix(c.cfg.OutputBucket, "/")
	outPath := fmt.Sprintf("%s/%s/%s/%s", bucket, req.VarLoc, version, key)
	runID := version + "/" + key // ledger object leaf: per (version, params)

	result := CreateJobResult{Hash: key, ModelVersion: version, OutputPath: outPath}

	cached, err := c.outputExists(ctx, outPath)
	if err != nil {
		return CreateJobResult{}, fmt.Errorf("checking output cache: %w", err)
	}
	if cached {
		result.Status = "cached"
		return result, nil
	}

	claimed, rec, err := c.claimRun(ctx, runID, req.JobName)
	if err != nil {
		return CreateJobResult{}, fmt.Errorf("claiming run: %w", err)
	}
	if !claimed {
		result.JobName = rec.JobName
		result.Attempts = rec.Attempts
		result.Status = "pending"
		return result, nil
	}

	// wait out any runner cold start on its own budget, so the /infer call
	// below is timed against inference alone.
	warmCtx, cancel := context.WithTimeout(ctx, warmupBudget)
	err = c.runner.Ready(warmCtx)
	cancel()
	if err != nil {
		c.releaseRun(ctx, runID)
		return CreateJobResult{}, fmt.Errorf("waiting for runner: %w", err)
	}

	inferCtx, cancel := context.WithTimeout(ctx, inferBudget)
	err = c.runner.Infer(inferCtx, inferRequest{
		VarLoc:        req.VarLoc,
		Version:       version,
		TreatmentArea: req.TreatmentArea,
		Hash:          key,
		OutputPath:    outPath,
	})
	cancel()
	if err != nil {
		c.releaseRun(ctx, runID)
		return CreateJobResult{}, fmt.Errorf("running inference: %w", err)
	}

	// output now exists; drop the claim so _runs/ doesn't accumulate.
	c.releaseRun(ctx, runID)

	result.JobName = req.JobName
	result.Attempts = rec.Attempts
	result.Status = "completed"
	return result, nil
}
