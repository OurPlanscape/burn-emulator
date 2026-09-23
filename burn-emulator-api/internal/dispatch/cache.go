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

const runStaleAfter = 35 * time.Minute

const maxClaimAttempts = 3

func CacheKey(req JobRequest) string {
	h := sha256.New()
	fmt.Fprintf(h, "%s|%s|%s", req.VarLoc, req.TreatmentArea, req.TreatmentAreaCRS)
	if req.IgnitionDensity != nil {
		fmt.Fprintf(h, "|%g", *req.IgnitionDensity)
	}
	return hex.EncodeToString(h.Sum(nil))
}

// claim ledger entry, stored as GCS object metadata.
type runRecord struct {
	Status    string
	JobName   string
	Attempts  int
	UpdatedAt time.Time
	Generation int64
}

func ledgerObjectName(key string) string {
	return "_runs/" + key
}

// claim a run atomically, using the ledger object's generation as a
// compare-and-swap: create it if absent, overwrite it if the claim is
// stale or not "running", and retry a lost race (412) up to
// maxClaimAttempts times.
func (c *Client) claimRun(ctx context.Context, key, jobName string) (bool, runRecord, error) {
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
			gen, putErr := c.putLedger(ctx, bucket, name, rec, 0)
			if putErr != nil {
				if isStatusCode(putErr, 412) {
					continue // lost the race; retry
				}
				return false, runRecord{}, fmt.Errorf("claiming run %s: %w", key, putErr)
			}
			rec.Generation = gen
			return true, rec, nil
		}

		rec := parseLedger(obj)
		if rec.Status == "running" && time.Since(rec.UpdatedAt) < runStaleAfter {
			return false, rec, nil
		}

		next := runRecord{Status: "running", JobName: jobName, Attempts: rec.Attempts + 1, UpdatedAt: time.Now()}
		gen, putErr := c.putLedger(ctx, bucket, name, next, obj.Generation)
		if putErr != nil {
			if isStatusCode(putErr, 412) {
				continue // someone else reclaimed it; retry
			}
			return false, runRecord{}, fmt.Errorf("reclaiming run %s: %w", key, putErr)
		}
		next.Generation = gen
		return true, next, nil
	}
	return false, runRecord{}, fmt.Errorf("claiming run %s: exceeded %d attempts under contention", key, maxClaimAttempts)
}

const releaseTimeout = 10 * time.Second

// report whether the ledger object is still the one we wrote (generation gen),
// i.e. our claim has not gone stale and been reclaimed by another run.
func (c *Client) ownsClaim(ctx context.Context, key string, gen int64) bool {
	bucket, err := c.outputBucketName()
	if err != nil {
		return false
	}
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), releaseTimeout)
	defer cancel()
	_, err = c.storage.Objects.Get(bucket, ledgerObjectName(key)).IfGenerationMatch(gen).Context(ctx).Do()
	if err != nil {
		if !isStatusCode(err, 404) && !isStatusCode(err, 412) {
			slog.Warn("failed to verify run claim", "key", key, "error", err)
		}
		return false
	}
	return true
}

// delete the ledger object, only if it is still the claim we wrote once
// a run finishes (output now exists) or fails (so the next run can claim it).
func (c *Client) releaseRun(ctx context.Context, key string, gen int64) {
	bucket, err := c.outputBucketName()
	if err != nil {
		slog.Warn("failed to clear run claim: bad output bucket", "key", key, "error", err)
		return
	}
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), releaseTimeout)
	defer cancel()
	err = c.storage.Objects.Delete(bucket, ledgerObjectName(key)).IfGenerationMatch(gen).Context(ctx).Do()
	if err != nil {
		if isStatusCode(err, 404) || isStatusCode(err, 412) {
			slog.Info("run claim no longer ours; leaving it", "key", key)
			return
		}
		slog.Warn("failed to clear run claim", "key", key, "error", err)
	}
}

// write rec as metadata on an empty object at bucket/name, conditioned on
// ifGenerationMatch (0 = must not exist, else = unchanged since read).
// Returns the new object generation, or a 412 error if the check fails.
func (c *Client) putLedger(ctx context.Context, bucket, name string, rec runRecord, ifGenerationMatch int64) (int64, error) {
	obj := &storage.Object{
		Name: name,
		Metadata: map[string]string{
			"status":     rec.Status,
			"job_name":   rec.JobName,
			"attempts":   strconv.Itoa(rec.Attempts),
			"updated_at": strconv.FormatInt(rec.UpdatedAt.Unix(), 10),
		},
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
