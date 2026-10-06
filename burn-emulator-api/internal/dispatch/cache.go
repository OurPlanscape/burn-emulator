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

// burn_emulator_runner_timeout (20m) + ~5m for container start and report
// delivery; keep above the runner timeout or healthy runs get duplicated.
const runStaleAfter = 25 * time.Minute

const maxClaimAttempts = 3

const releaseTimeout = 10 * time.Second

const (
	claimPrefix  = "_claims/"
	reportPrefix = "_reports/"
)

// claim entry, stored as GCS object metadata.
type claimRecord struct {
	Status     string // "running" | "failed"
	JobName    string
	Attempts   int
	UpdatedAt  time.Time
	Generation int64
	Error      string
}

func CacheKey(req JobRequest) string {
	h := sha256.New()
	fmt.Fprintf(h, "%s|%s", req.VarLoc, req.TreatmentArea)
	if req.IgnitionDensity != nil {
		fmt.Fprintf(h, "|%g", *req.IgnitionDensity)
	}
	key := hex.EncodeToString(h.Sum(nil))
	if req.Backend == BackendPT {
		return key + "1"
	}
	return key + "0"
}

// claim a run atomically using the claim object's generation
func (c *Client) claimRun(ctx context.Context, key, jobName string) (bool, claimRecord, error) {
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
			rec := claimRecord{Status: "running", JobName: jobName, Attempts: 1, UpdatedAt: time.Now()}
			gen, putErr := c.putClaim(ctx, bucket, name, rec, 0)
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
		if rec.Status == "running" && time.Since(rec.UpdatedAt) < runStaleAfter {
			return false, rec, nil
		}

		next := claimRecord{Status: "running", JobName: jobName, Attempts: rec.Attempts + 1, UpdatedAt: time.Now()}
		gen, putErr := c.putClaim(ctx, bucket, name, next, obj.Generation)
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

// report whether the claim object is still the one the api wrote
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

// delete the claim object, only if it is still the claim the api wrote once
// a run finishes (output now exists) or fails (so the next run can claim it).
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

// GCS uploads are atomic (an object only appears once its upload completes)
func (c *Client) clearFinishedClaim(ctx context.Context, key string) {
	rec, found, err := c.readClaim(ctx, key)
	if err != nil || !found || rec.Status != "running" {
		return
	}
	c.releaseRun(ctx, key, rec.Generation)
}

// mark the api's claim (generation gen) "failed" so GET reports it until the
// next POST reclaims it. A claim reclaimed by another run is left alone.
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
	rec.Status = "failed"
	rec.Error = reason
	rec.UpdatedAt = time.Now()
	if _, err := c.putClaim(ctx, bucket, name, rec, gen); err != nil && !isStatusCode(err, 412) {
		return fmt.Errorf("marking run %s failed: %w", key, err)
	}
	return nil
}

func claimObjectName(key string) string {
	return claimPrefix + key
}

// write rec as metadata on an empty object at bucket/name, conditioned on
// ifGenerationMatch (0 = must not exist, else = unchanged since read).
// Returns the new object generation, or a 412 error if the check fails.
func (c *Client) putClaim(ctx context.Context, bucket, name string, rec claimRecord, ifGenerationMatch int64) (int64, error) {
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
	written, err := c.storage.Objects.Insert(bucket, obj).
		Media(strings.NewReader("")).
		IfGenerationMatch(ifGenerationMatch).
		Context(ctx).
		Do()
	if err != nil {
		return 0, err
	}
	return written.Generation, nil
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
	bucket, prefix, err := parseGCSPath(gcsPath)
	if err != nil {
		return false, err
	}
	resp, err := c.storage.Objects.List(bucket).Prefix(prefix + "/").MaxResults(1).Context(ctx).Do()
	if err != nil {
		return false, fmt.Errorf("listing gs://%s/%s: %w", bucket, prefix, err)
	}
	return len(resp.Items) > 0, nil
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
