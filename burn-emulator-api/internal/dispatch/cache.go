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

// how long a "running" claim blocks a retry.
const runStaleAfter = 3 * time.Minute

// max retries when losing a claim race.
const maxClaimAttempts = 3

// hash varloc + treatment area into the cache key.
func CacheKey(req JobRequest) string {
	h := sha256.New()
	fmt.Fprintf(h, "%s|%s", req.VarLoc, req.TreatmentArea)
	return hex.EncodeToString(h.Sum(nil))
}

// claim ledger entry, stored as GCS object metadata.
type runRecord struct {
	Status    string
	JobName   string
	Attempts  int
	UpdatedAt time.Time
}

// build the ledger object name for a key.
func ledgerObjectName(key string) string {
	return "_runs/" + key
}

// claim a run atomically, using the ledger object's generation as a
// compare-and-swap: create it if absent, overwrite it if the claim is
// stale or not "running", and retry a lost race (412) up to
// maxClaimAttempts times.
func (c *Client) claimRun(ctx context.Context, key, jobName string) (claimed bool, existing runRecord, err error) {
	bucket, err := c.outputBucketName()
	if err != nil {
		return false, runRecord{}, err
	}
	name := ledgerObjectName(key)

	for attempt := 0; attempt < maxClaimAttempts; attempt++ {
		obj, getErr := c.storage.Objects.Get(bucket, name).Context(ctx).Do()
		if getErr != nil {
			if !isStatusCode(getErr, 404) {
				return false, runRecord{}, fmt.Errorf("reading run ledger %s: %w", key, getErr)
			}
			rec := runRecord{Status: "running", JobName: jobName, Attempts: 1, UpdatedAt: time.Now()}
			if putErr := c.putLedger(ctx, bucket, name, rec, 0); putErr != nil {
				if isStatusCode(putErr, 412) {
					continue // lost the race; retry
				}
				return false, runRecord{}, fmt.Errorf("claiming run %s: %w", key, putErr)
			}
			return true, rec, nil
		}

		rec := parseLedger(obj)
		if rec.Status == "running" && time.Since(rec.UpdatedAt) < runStaleAfter {
			return false, rec, nil
		}

		next := runRecord{Status: "running", JobName: jobName, Attempts: rec.Attempts + 1, UpdatedAt: time.Now()}
		if putErr := c.putLedger(ctx, bucket, name, next, obj.Generation); putErr != nil {
			if isStatusCode(putErr, 412) {
				continue // someone else reclaimed it; retry
			}
			return false, runRecord{}, fmt.Errorf("reclaiming run %s: %w", key, putErr)
		}
		return true, next, nil
	}
	return false, runRecord{}, fmt.Errorf("claiming run %s: exceeded %d attempts under contention", key, maxClaimAttempts)
}

// delete the ledger object: called once a run finishes (output now exists) or
// fails (so the next request can retry without waiting out runStaleAfter).
func (c *Client) releaseRun(ctx context.Context, key string) {
	bucket, err := c.outputBucketName()
	if err != nil {
		slog.Warn("failed to clear run claim: bad output bucket", "key", key, "error", err)
		return
	}
	if err := c.storage.Objects.Delete(bucket, ledgerObjectName(key)).Context(ctx).Do(); err != nil {
		slog.Warn("failed to clear run claim", "key", key, "error", err)
	}
}

// write rec as metadata on an empty object at bucket/name, conditioned on
// ifGenerationMatch (0 = must not exist, else = unchanged since read).
// Returns 412 if the check fails.
func (c *Client) putLedger(ctx context.Context, bucket, name string, rec runRecord, ifGenerationMatch int64) error {
	obj := &storage.Object{
		Name: name,
		Metadata: map[string]string{
			"status":     rec.Status,
			"job_name":   rec.JobName,
			"attempts":   strconv.Itoa(rec.Attempts),
			"updated_at": strconv.FormatInt(rec.UpdatedAt.Unix(), 10),
		},
	}
	_, err := c.storage.Objects.Insert(bucket, obj).
		Media(strings.NewReader("")).
		IfGenerationMatch(ifGenerationMatch).
		Context(ctx).
		Do()
	return err
}

// decode a runRecord from a ledger object's metadata.
func parseLedger(obj *storage.Object) runRecord {
	attempts, _ := strconv.Atoi(obj.Metadata["attempts"])
	updatedUnix, _ := strconv.ParseInt(obj.Metadata["updated_at"], 10, 64)
	return runRecord{
		Status:    obj.Metadata["status"],
		JobName:   obj.Metadata["job_name"],
		Attempts:  attempts,
		UpdatedAt: time.Unix(updatedUnix, 0),
	}
}

// report whether a prior run already wrote output under gcsPath.
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

// extract the bare bucket name from the gs://<bucket> config.
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

// split a gs://<bucket>/<object> URI into bucket and object.
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
