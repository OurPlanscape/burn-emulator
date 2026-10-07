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

// claim + trigger timeout, detached from the caller's context
const claimTriggerTimeout = 90 * time.Second

// max time CreateJob runs past its caller's deadline
const DetachedBudget = claimTriggerTimeout + releaseTimeout

var ErrJobNotFound = errors.New("job not found")

// sha256 hex + one backend bit: 0 DL, 1 PT
var validHash = regexp.MustCompile(`^[0-9a-f]{64}[01]$`)

// must match BACKENDS in burn-emulator-model's constants.py
const (
	BackendDL = "DL" // deep learning emulator, GPU runner job
	BackendPT = "PT" // pyretechnics, CPU-only runner job
)

type JobRequest struct {
	TreatmentArea   string
	VarLoc          string // optional; picked from the treatment area when empty
	JobName         string
	IgnitionDensity *float64
	Backend         string // BackendDL | BackendPT
}

// model_version segment of PT jobs
const ptModelVersion = "pt"

type JobID struct {
	InputsVersion string // fuels (baseline + legalmax), topo and varlocs, published together
	VarLoc        string // see selectVarLoc
	ModelVersion  string // <models>/<varloc>/current for DL, ptModelVersion for PT
	Hash          string // cache key for the request parameters
}

// <inputs_version>/<varloc>/<model_version>/<hash>
func (id JobID) Path() string {
	return strings.Join([]string{id.InputsVersion, id.VarLoc, id.ModelVersion, id.Hash}, "/")
}

func (id JobID) Backend() string {
	if strings.HasSuffix(id.Hash, "1") {
		return BackendPT
	}
	return BackendDL
}

func (id JobID) valid() bool {
	return validVersion.MatchString(id.InputsVersion) &&
		validVersion.MatchString(id.VarLoc) &&
		validVersion.MatchString(id.ModelVersion) &&
		validHash.MatchString(id.Hash)
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
	inputsVersion, err := c.inputsVersions.resolve(ctx, "")
	if err != nil {
		return JobResult{}, fmt.Errorf("resolving inputs version: %w", err)
	}
	varLoc, modelVersion, backend, err := c.selectVarLoc(ctx, inputsVersion, req.TreatmentArea, req.VarLoc, req.Backend)
	if err != nil {
		return JobResult{}, err
	}
	req.Backend = backend

	id := JobID{
		InputsVersion: inputsVersion,
		VarLoc:        varLoc,
		ModelVersion:  modelVersion,
		Hash:          CacheKey(req),
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

	treatmentAreaPath := outPath + "/" + treatmentAreaObject
	if err := c.writeTreatmentArea(detachedCtx, treatmentAreaPath, req.TreatmentArea); err != nil {
		c.releaseRun(ctx, runObject, rec.Generation)
		return JobResult{}, err
	}

	err = c.runner.Trigger(detachedCtx, inferRequest{
		InputsVersion:     inputsVersion,
		VarLoc:            varLoc,
		ModelVersion:      modelVersion,
		TreatmentAreaPath: treatmentAreaPath,
		IgnitionDensity:   req.IgnitionDensity,
		Backend:           req.Backend,
		Hash:              id.Hash,
		OutputPath:        outPath,
		ReportPath:        reportPath,
		ClaimGeneration:   rec.Generation,
	})
	if err != nil {
		if isRejected(err) {
			c.releaseRun(ctx, runObject, rec.Generation)
		} else {
			// an execution may have started; the claim is kept and goes stale after runStaleAfter
			slog.Warn("runner trigger outcome unknown; keeping claim", "run", runObject, "error", err)
		}
		return JobResult{}, fmt.Errorf("triggering inference: %w", err)
	}

	result.JobName = req.JobName
	result.Attempts = rec.Attempts
	result.Status = "pending"
	return result, nil
}

// ErrJobNotFound: malformed id, or neither output nor claim
func (c *Client) GetJob(ctx context.Context, id JobID) (JobResult, error) {
	if !id.valid() {
		return JobResult{}, ErrJobNotFound
	}
	outPath := strings.TrimSuffix(c.cfg.OutputBucket, "/") + "/" + id.Path()
	result := JobResult{ID: id, OutputPath: outPath}

	cached, err := c.outputExists(ctx, outPath)
	if err != nil {
		return JobResult{}, fmt.Errorf("checking output: %w", err)
	}
	if cached {
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
