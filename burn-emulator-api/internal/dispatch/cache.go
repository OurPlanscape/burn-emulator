package dispatch

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"log/slog"
	"strconv"
	"strings"
	"time"

	storage "google.golang.org/api/storage/v1"
)

const (
	runStaleAfter     = 30 * time.Minute  // running
	queueStaleAfter   = 2 * runStaleAfter // queued (metal): waiting for a runner loop
	stagingStaleAfter = DetachedBudget    // staging (metal): not queued by CreateJob
	maxClaimAttempts  = 3
	releaseTimeout    = 10 * time.Second
)

const (
	claimPrefix  = "_claims/"
	reportPrefix = "_reports/"
)

const (
	outputObject = "output.tif" // its existence is the cache hit
	metaObject   = "meta.geojson"
)

const (
	statusStaging = "staging" // metal: held by CreateJob until meta.geojson is written
	statusQueued  = "queued"
	statusRunning = "running"
	statusFailed  = "failed"
)

// a claim in one of these expires without a report, see claimRecord.stale
func isActive(status string) bool {
	return status == statusStaging || status == statusQueued || status == statusRunning
}

// stored as claim object metadata; the runner's claim loop adds runner, dropped on the next API write
type claimRecord struct {
	Status     string // statusStaging | statusQueued | statusRunning | statusFailed
	JobName    string
	Attempts   int
	UpdatedAt  time.Time
	Generation int64
	Error      string
}

func CacheKey(req JobRequest) string {
	h := sha256.New()
	fmt.Fprint(h, req.TreatmentArea)
	if req.IgnitionDensity != nil {
		fmt.Fprintf(h, "|%g", *req.IgnitionDensity)
	}
	key := hex.EncodeToString(h.Sum(nil))
	if req.Backend == BackendPT {
		return key + "1"
	}
	return key + "0"
}

// claims a run with a generation precondition on the claim object; metal mode then hands it over with queueRun
func (c *Client) claimRun(ctx context.Context, key, jobName, status, body string) (bool, claimRecord, error) {
	bucket, err := c.outputBucketName()
	if err != nil {
		return false, claimRecord{}, err
	}
	name := claimObjectName(key)

	for attempt := 0; attempt < maxClaimAttempts; attempt++ {
		obj, getErr := c.storage.Objects.Get(bucket, name).Context(ctx).Do()
		if getErr != nil {
			if !isStatusCode(getErr, 404) {
				return false, claimRecord{}, fmt.Errorf("reading run claim %s: %w", key, getErr)
			}
			rec := claimRecord{Status: status, JobName: jobName, Attempts: 1, UpdatedAt: time.Now()}
			gen, putErr := c.putClaim(ctx, bucket, name, rec, body, 0)
			if putErr != nil {
				if isStatusCode(putErr, 412) {
					continue // lost the race; retry
				}
				return false, claimRecord{}, fmt.Errorf("claiming run %s: %w", key, putErr)
			}
			rec.Generation = gen
			return true, rec, nil
		}

		rec := parseClaim(obj)
		if isActive(rec.Status) && !rec.stale() {
			return false, rec, nil
		}

		next := claimRecord{Status: status, JobName: jobName, Attempts: rec.Attempts + 1, UpdatedAt: time.Now()}
		gen, putErr := c.putClaim(ctx, bucket, name, next, body, obj.Generation)
		if putErr != nil {
			if isStatusCode(putErr, 412) {
				continue // someone else reclaimed it; retry
			}
			return false, claimRecord{}, fmt.Errorf("reclaiming run %s: %w", key, putErr)
		}
		next.Generation = gen
		return true, next, nil
	}
	return false, claimRecord{}, fmt.Errorf("claiming run %s: exceeded %d attempts under contention", key, maxClaimAttempts)
}

func (c *Client) readClaim(ctx context.Context, key string) (claimRecord, bool, error) {
	bucket, err := c.outputBucketName()
	if err != nil {
		return claimRecord{}, false, err
	}
	obj, err := c.storage.Objects.Get(bucket, claimObjectName(key)).Context(ctx).Do()
	if err != nil {
		if isStatusCode(err, 404) {
			return claimRecord{}, false, nil
		}
		return claimRecord{}, false, fmt.Errorf("reading run claim %s: %w", key, err)
	}
	return parseClaim(obj), true, nil
}

// whether the claim object is still generation gen
func (c *Client) ownsClaim(ctx context.Context, key string, gen int64) bool {
	bucket, err := c.outputBucketName()
	if err != nil {
		return false
	}
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), releaseTimeout)
	defer cancel()
	_, err = c.storage.Objects.Get(bucket, claimObjectName(key)).IfGenerationMatch(gen).Context(ctx).Do()
	if err != nil {
		if !isStatusCode(err, 404) && !isStatusCode(err, 412) {
			slog.Warn("failed to verify run claim", "key", key, "error", err)
		}
		return false
	}
	return true
}

// deletes the claim object if it is still generation gen
func (c *Client) releaseRun(ctx context.Context, key string, gen int64) {
	bucket, err := c.outputBucketName()
	if err != nil {
		slog.Warn("failed to clear run claim: bad output bucket", "key", key, "error", err)
		return
	}
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), releaseTimeout)
	defer cancel()
	err = c.storage.Objects.Delete(bucket, claimObjectName(key)).IfGenerationMatch(gen).Context(ctx).Do()
	if err != nil {
		if isStatusCode(err, 404) || isStatusCode(err, 412) {
			slog.Info("run claim reclaimed or already cleared; leaving it", "key", key)
			return
		}
		slog.Warn("failed to clear run claim", "key", key, "error", err)
	}
}

// releases an active claim whose output exists
func (c *Client) clearFinishedClaim(ctx context.Context, key string) {
	rec, found, err := c.readClaim(ctx, key)
	if err != nil || !found || !isActive(rec.Status) {
		return
	}
	c.releaseRun(ctx, key, rec.Generation)
}

// marks the claim failed if it is still generation gen
func (c *Client) failRun(ctx context.Context, key string, gen int64, reason string) error {
	bucket, err := c.outputBucketName()
	if err != nil {
		return err
	}
	name := claimObjectName(key)
	obj, err := c.storage.Objects.Get(bucket, name).IfGenerationMatch(gen).Context(ctx).Do()
	if err != nil {
		if isStatusCode(err, 404) || isStatusCode(err, 412) {
			return nil
		}
		return fmt.Errorf("reading run claim %s: %w", key, err)
	}
	rec := parseClaim(obj)
	rec.Status = statusFailed
	rec.Error = reason
	rec.UpdatedAt = time.Now()
	if _, err := c.putClaim(ctx, bucket, name, rec, "", gen); err != nil && !isStatusCode(err, 412) {
		return fmt.Errorf("marking run %s failed: %w", key, err)
	}
	return nil
}

// hands a claim taken as staging to the runner's claim loop, if it is still generation rec.Generation
func (c *Client) queueRun(ctx context.Context, key string, rec claimRecord, body string) error {
	bucket, err := c.outputBucketName()
	if err != nil {
		return err
	}
	rec.Status = statusQueued
	rec.UpdatedAt = time.Now()
	if _, err := c.putClaim(ctx, bucket, claimObjectName(key), rec, body, rec.Generation); err != nil {
		return fmt.Errorf("queueing run %s: %w", key, err)
	}
	return nil
}

func claimObjectName(key string) string {
	return claimPrefix + key
}

// ifGenerationMatch 0: the object must not exist. Returns the new generation; 412 on a failed precondition.
func (c *Client) putClaim(ctx context.Context, bucket, name string, rec claimRecord, body string, ifGenerationMatch int64) (int64, error) {
	obj := &storage.Object{
		Name: name,
		Metadata: map[string]string{
			"status":     rec.Status,
			"job_name":   rec.JobName,
			"attempts":   strconv.Itoa(rec.Attempts),
			"updated_at": strconv.FormatInt(rec.UpdatedAt.Unix(), 10),
		},
	}
	if rec.Error != "" {
		obj.Metadata["error"] = rec.Error
	}
	if body != "" {
		obj.ContentType = "application/json"
	}
	written, err := c.storage.Objects.Insert(bucket, obj).
		Media(strings.NewReader(body)).
		IfGenerationMatch(ifGenerationMatch).
		Context(ctx).
		Do()
	if err != nil {
		return 0, err
	}
	return written.Generation, nil
}

// updated_at is when the claim was taken or queued, or when its run started
func (r claimRecord) stale() bool {
	switch r.Status {
	case statusStaging:
		return time.Since(r.UpdatedAt) >= stagingStaleAfter
	case statusRunning:
		return time.Since(r.UpdatedAt) >= runStaleAfter
	case statusQueued:
		return time.Since(r.UpdatedAt) >= queueStaleAfter
	}
	return false
}

func parseClaim(obj *storage.Object) claimRecord {
	attempts, _ := strconv.Atoi(obj.Metadata["attempts"])
	updatedUnix, _ := strconv.ParseInt(obj.Metadata["updated_at"], 10, 64)
	return claimRecord{
		Status:     obj.Metadata["status"],
		JobName:    obj.Metadata["job_name"],
		Attempts:   attempts,
		UpdatedAt:  time.Unix(updatedUnix, 0),
		Generation: obj.Generation,
		Error:      obj.Metadata["error"],
	}
}

func (c *Client) outputExists(ctx context.Context, gcsPath string) (bool, error) {
	bucket, name, err := parseGCSPath(gcsPath)
	if err != nil {
		return false, err
	}
	_, err = c.storage.Objects.Get(bucket, name).Context(ctx).Do()
	if isStatusCode(err, 404) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("checking gs://%s/%s: %w", bucket, name, err)
	}
	return true, nil
}

func (c *Client) writeTreatmentArea(ctx context.Context, gcsPath, geojson string) error {
	bucket, name, err := parseGCSPath(gcsPath)
	if err != nil {
		return err
	}
	obj := &storage.Object{Name: name, ContentType: "application/geo+json"}
	_, err = c.storage.Objects.Insert(bucket, obj).Media(strings.NewReader(geojson)).Context(ctx).Do()
	if err != nil {
		return fmt.Errorf("writing treatment area %s: %w", gcsPath, err)
	}
	return nil
}

func (c *Client) deleteOutput(ctx context.Context, gcsPath string) {
	bucket, prefix, err := parseGCSPath(gcsPath)
	if err != nil {
		slog.Warn("failed to clean up output: bad output path", "path", gcsPath, "error", err)
		return
	}
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), releaseTimeout)
	defer cancel()

	pageToken := ""
	for {
		resp, err := c.storage.Objects.List(bucket).Prefix(prefix + "/").PageToken(pageToken).Context(ctx).Do()
		if err != nil {
			slog.Warn("failed to list output for cleanup", "path", gcsPath, "error", err)
			return
		}
		for _, obj := range resp.Items {
			if err := c.storage.Objects.Delete(bucket, obj.Name).Context(ctx).Do(); err != nil {
				slog.Warn("failed to delete output object", "object", obj.Name, "error", err)
			}
		}
		if resp.NextPageToken == "" {
			return
		}
		pageToken = resp.NextPageToken
	}
}

func (c *Client) outputBucketName() (string, error) {
	const prefix = "gs://"
	if !strings.HasPrefix(c.cfg.OutputBucket, prefix) {
		return "", fmt.Errorf("output bucket %q is not a gs:// path", c.cfg.OutputBucket)
	}
	bucket := strings.Trim(strings.TrimPrefix(c.cfg.OutputBucket, prefix), "/")
	if bucket == "" {
		return "", fmt.Errorf("output bucket %q has no bucket name", c.cfg.OutputBucket)
	}
	return bucket, nil
}

func parseGCSPath(uri string) (bucket, object string, err error) {
	const prefix = "gs://"
	if !strings.HasPrefix(uri, prefix) {
		return "", "", fmt.Errorf("not a gs:// path: %s", uri)
	}
	parts := strings.SplitN(strings.TrimPrefix(uri, prefix), "/", 2)
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return "", "", fmt.Errorf("malformed gs:// path: %s", uri)
	}
	return parts[0], parts[1], nil
}
