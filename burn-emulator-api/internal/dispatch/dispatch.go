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

const (
	claimTriggerTimeout = 90 * time.Second                     // detached from the caller's context
	DetachedBudget      = claimTriggerTimeout + releaseTimeout // max time CreateJob runs past its caller's deadline
)

var ErrJobNotFound = errors.New("job not found")

var validHash = regexp.MustCompile(`^[0-9a-f]{64}[01]$`) // sha256 hex + backend bit: 0 DL, 1 PT

// model BACKENDS in constants.py
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

const ptModelVersion = "pt" // model_version segment of PT jobs, pt-<model_version> with a bundle

type JobID struct {
	InputsVersion string // fuels (baseline + legalmax), topo and varlocs, published together
	VarLoc        string // see selectVarLoc
	ModelVersion  string // see jobModelVersion
	Hash          string // cache key for the request parameters
}

// DL: the bundle's model_version; PT: pt, or pt-<model_version> with a bundle
func jobModelVersion(backend, modelVersion string) string {
	switch {
	case backend == BackendDL:
		return modelVersion
	case modelVersion == "":
		return ptModelVersion
	default:
		return ptModelVersion + "-" + modelVersion
	}
}

// the bundle's model_version, "" for PT without one
func (id JobID) BundleVersion() string {
	if id.Backend() == BackendDL {
		return id.ModelVersion
	}
	v := strings.TrimPrefix(id.ModelVersion, ptModelVersion)
	return strings.TrimPrefix(v, "-")
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
	OutputDir  string // gs:// <output_uri>/<job_id>
	OutputPath string // <OutputDir>/output.tif
	OutputMeta string // <OutputDir>/meta.geojson
	Error      string // runner error, set when Status is "failed"
}

func (c *Client) newJobResult(id JobID) JobResult {
	dir := strings.TrimSuffix(c.cfg.OutputBucket, "/") + "/" + id.Path()
	return JobResult{
		ID:         id,
		OutputDir:  dir,
		OutputPath: dir + "/" + outputObject,
		OutputMeta: dir + "/" + metaObject,
	}
}

func (c *Client) CreateJob(ctx context.Context, req JobRequest) (JobResult, error) {
	area, err := treatmentArea(req.TreatmentArea)
	if err != nil {
		return JobResult{}, fmt.Errorf("%w: %v", ErrInvalidTreatmentArea, err)
	}
	inputsVersion, err := c.inputsVersions.resolve(ctx, "")
	if err != nil {
		return JobResult{}, fmt.Errorf("resolving inputs version: %w", err)
	}
	varLoc, modelVersion, backend, err := c.selectVarLoc(ctx, inputsVersion, area, req.VarLoc, req.Backend)
	if err != nil {
		return JobResult{}, err
	}
	req.Backend = backend

	id := JobID{
		InputsVersion: inputsVersion,
		VarLoc:        varLoc,
		ModelVersion:  jobModelVersion(backend, modelVersion),
		Hash:          CacheKey(req),
	}
	runObject := id.Path()
	reportPath := strings.TrimSuffix(c.cfg.OutputBucket, "/") + "/" + reportPrefix + runObject

	result := c.newJobResult(id)

	cached, err := c.outputExists(ctx, result.OutputPath)
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

	if err := c.writeTreatmentArea(detachedCtx, result.OutputMeta, workingGeoJSON(area)); err != nil {
		c.releaseRun(ctx, runObject, rec.Generation)
		return JobResult{}, err
	}

	var bundleURI string
	if modelVersion != "" {
		bundleURI = strings.TrimSuffix(c.cfg.ModelsURI, "/") + "/" + varLoc + "/" + modelVersion
	}
	err = c.runner.Trigger(detachedCtx, inferRequest{
		InputsVersion:   inputsVersion,
		VarLoc:          varLoc,
		ModelVersion:    modelVersion,
		BundleURI:       bundleURI,
		OutputMeta:      result.OutputMeta,
		IgnitionDensity: req.IgnitionDensity,
		Backend:         req.Backend,
		Hash:            id.Hash,
		OutputPath:      result.OutputPath,
		ReportPath:      reportPath,
		ClaimGeneration: rec.Generation,
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
	result := c.newJobResult(id)

	cached, err := c.outputExists(ctx, result.OutputPath)
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
