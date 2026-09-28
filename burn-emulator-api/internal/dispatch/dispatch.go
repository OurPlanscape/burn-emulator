package dispatch

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"regexp"
	"strings"
	"time"
)

// how long the claim + runner job trigger get, detached from the caller so a
// hangup can't leave a claim with no job behind it, or a job with no claim.
const claimTriggerTimeout = 90 * time.Second

// the most CreateJob can run past its caller's deadline: the detached claim +
// trigger, then a detached release if the trigger fails.
const DetachedBudget = claimTriggerTimeout + releaseTimeout

var ErrJobNotFound = errors.New("job not found")

var validHash = regexp.MustCompile(`^[0-9a-f]{64}$`)

type JobRequest struct {
	TreatmentArea    string
	TreatmentAreaCRS string
	VarLoc           string
	JobName          string
	IgnitionDensity  *float64
}

type JobID struct {
	VarLoc       string
	ModelVersion string
	DataVersion  string // fuels (baseline + legalmax) and topo, published together
	Hash         string // cache key for the request parameters
}

// <varloc>/<model_version>/<data_version>/<hash>
func (id JobID) Path() string {
	return strings.Join([]string{id.VarLoc, id.ModelVersion, id.DataVersion, id.Hash}, "/")
}

func (id JobID) valid() bool {
	for _, seg := range []string{id.VarLoc, id.ModelVersion, id.DataVersion} {
		if !validVersion.MatchString(seg) {
			return false
		}
	}
	return validHash.MatchString(id.Hash)
}

type JobResult struct {
	ID         JobID
	JobName    string
	Status     string // POST: "cached" | "pending"; GET: "cached" | "pending" | "failed"
	Attempts   int
	OutputPath string
	Error      string // runner error, set when Status is "failed"
}

func (c *Client) CreateJob(ctx context.Context, req JobRequest) (JobResult, error) {
	modelVersion, err := c.modelVersions.resolve(ctx, req.VarLoc)
	if err != nil {
		return JobResult{}, fmt.Errorf("resolving model version for %s: %w", req.VarLoc, err)
	}
	dataVersion, err := c.dataVersions.resolve(ctx, "")
	if err != nil {
		return JobResult{}, fmt.Errorf("resolving data version: %w", err)
	}

	id := JobID{
		VarLoc:       req.VarLoc,
		ModelVersion: modelVersion,
		DataVersion:  dataVersion,
		Hash:         CacheKey(req),
	}
	runObject := id.Path()
	bucket := strings.TrimSuffix(c.cfg.OutputBucket, "/")
	outPath := bucket + "/" + runObject
	reportPath := bucket + "/" + reportPrefix + runObject

	result := JobResult{ID: id, OutputPath: outPath}

	cached, err := c.outputExists(ctx, outPath)
	if err != nil {
		return JobResult{}, fmt.Errorf("checking output cache: %w", err)
	}
	if cached {
		c.clearFinishedClaim(ctx, runObject)
		result.Status = "cached"
		return result, nil
	}

	detachedCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), claimTriggerTimeout)
	defer cancel()

	claimed, rec, err := c.claimRun(detachedCtx, runObject, req.JobName)
	if err != nil {
		return JobResult{}, fmt.Errorf("claiming run: %w", err)
	}
	if !claimed {
		result.JobName = rec.JobName
		result.Attempts = rec.Attempts
		result.Status = "pending"
		return result, nil
	}

	err = c.runner.Trigger(detachedCtx, inferRequest{
		VarLoc:           req.VarLoc,
		ModelVersion:     modelVersion,
		DataVersion:      dataVersion,
		TreatmentArea:    req.TreatmentArea,
		TreatmentAreaCRS: req.TreatmentAreaCRS,
		IgnitionDensity:  req.IgnitionDensity,
		Hash:             id.Hash,
		OutputPath:       outPath,
		ReportPath:       reportPath,
		ClaimGeneration:  rec.Generation,
	})
	if err != nil {
		if isRejected(err) {
			c.releaseRun(ctx, runObject, rec.Generation)
		} else {
			// an execution may have started without the api hearing back, so keep the
			// claim rather than risk a duplicate run; if nothing started it goes
			// stale after runStaleAfter and GET reports it failed.
			slog.Warn("runner trigger outcome unknown; keeping claim", "run", runObject, "error", err)
		}
		return JobResult{}, fmt.Errorf("triggering inference: %w", err)
	}

	result.JobName = req.JobName
	result.Attempts = rec.Attempts
	result.Status = "pending"
	return result, nil
}

// read-only status of a run. ErrJobNotFound covers a malformed id, and a run
// with neither output nor claim (never started, or its claim was cleared).
func (c *Client) GetJob(ctx context.Context, id JobID) (JobResult, error) {
	if !id.valid() {
		return JobResult{}, ErrJobNotFound
	}
	outPath := strings.TrimSuffix(c.cfg.OutputBucket, "/") + "/" + id.Path()
	result := JobResult{ID: id, OutputPath: outPath}

	done, err := c.outputExists(ctx, outPath)
	if err != nil {
		return JobResult{}, fmt.Errorf("checking output: %w", err)
	}
	if done {
		c.clearFinishedClaim(ctx, id.Path())
		result.Status = "cached"
		return result, nil
	}

	rec, found, err := c.readClaim(ctx, id.Path())
	if err != nil {
		return JobResult{}, err
	}
	if !found {
		return JobResult{}, ErrJobNotFound
	}
	result.JobName = rec.JobName
	result.Attempts = rec.Attempts

	switch {
	case rec.Status == "failed":
		result.Status = "failed"
		result.Error = rec.Error
	case time.Since(rec.UpdatedAt) >= runStaleAfter:
		result.Status = "failed"
		result.Error = "run did not report back before its claim went stale"
	default:
		result.Status = "pending"
	}
	return result, nil
}
