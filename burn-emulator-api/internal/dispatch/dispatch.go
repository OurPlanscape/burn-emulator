package dispatch

import (
	"context"
	"fmt"
	"strings"
	"time"
)

// how long a runner job execution (trigger + wait) gets, comfortably above the
// job's own template timeout so the job's timeout fires and reports cleanly first.
const runBudget = 25 * time.Minute

type JobRequest struct {
	TreatmentArea    string
	TreatmentAreaCRS string
	VarLoc           string
	JobName          string
	IgnitionDensity  *float64
}

type CreateJobResult struct {
	JobName      string
	Hash         string // cache key for the request parameters
	ModelVersion string
	FuelsVersion string
	TopoVersion  string
	Status       string // "cached" | "pending" | "completed"
	Attempts     int
	OutputPath   string
}

func (c *Client) CreateJob(ctx context.Context, req JobRequest) (CreateJobResult, error) {
	version, err := c.versions.resolve(ctx, req.VarLoc)
	if err != nil {
		return CreateJobResult{}, fmt.Errorf("resolving model version for %s: %w", req.VarLoc, err)
	}
	fuelsVersion, err := c.inputVersions.resolve(ctx, "fuels")
	if err != nil {
		return CreateJobResult{}, fmt.Errorf("resolving fuels version: %w", err)
	}
	topoVersion, err := c.inputVersions.resolve(ctx, "topo")
	if err != nil {
		return CreateJobResult{}, fmt.Errorf("resolving topo version: %w", err)
	}

	key := CacheKey(req)
	bucket := strings.TrimSuffix(c.cfg.OutputBucket, "/")
	outPath := fmt.Sprintf("%s/%s/%s/%s-%s/%s", bucket, req.VarLoc, version, fuelsVersion, topoVersion, key)
	runID := version + "/" + fuelsVersion + "-" + topoVersion + "/" + key // ledger object leaf: per (version, inputs, params)

	result := CreateJobResult{
		Hash:         key,
		ModelVersion: version,
		FuelsVersion: fuelsVersion,
		TopoVersion:  topoVersion,
		OutputPath:   outPath,
	}

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

	runCtx, cancel := context.WithTimeout(ctx, runBudget)
	err = c.runner.Run(runCtx, inferRequest{
		VarLoc:           req.VarLoc,
		Version:          version,
		FuelsVersion:     fuelsVersion,
		TopoVersion:      topoVersion,
		TreatmentArea:    req.TreatmentArea,
		TreatmentAreaCRS: req.TreatmentAreaCRS,
		IgnitionDensity:  req.IgnitionDensity,
		Hash:             key,
		OutputPath:       outPath,
	})
	cancel()
	if err != nil {
		c.deleteOutput(ctx, outPath)
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
