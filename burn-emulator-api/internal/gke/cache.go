package k8s

import (
	"context"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"fmt"
	"log/slog"
	"strings"
	"time"
)

// how long a "running" claim blocks a retry if the job
// that made it never reports back (e.g. it crashed before writing output).
const runStaleAfter = 30 * time.Minute

// how long a run's state survives in Redis after its last
// update, so old records don't accumulate forever.
const runRecordTTL = 24 * time.Hour

//go:embed claim.lua
var claimScript string

func CacheKey(req JobRequest) string {
	h := sha256.New()
	fmt.Fprintf(h, "%s|%s|%.6f|%.6f|%.6f", req.Variation, req.TreatmentArea, req.TreatmentBuff, req.TreatmentSeed, req.IgnitionDensity)
	return hex.EncodeToString(h.Sum(nil))
}

type runRecord struct {
	Status   string
	JobName  string
	Attempts int
}

func (c *Client) claimRun(ctx context.Context, key, jobName string) (claimed bool, existing runRecord, err error) {
	now := time.Now().Unix()
	res, err := c.redis.Eval(ctx, claimScript,
		[]string{runKey(key)},
		jobName, now, int(runStaleAfter.Seconds()), int(runRecordTTL.Seconds()),
	).Result()
	if err != nil {
		return false, runRecord{}, fmt.Errorf("claiming run %s: %w", key, err)
	}

	fields, ok := res.([]interface{})
	if !ok || len(fields) != 4 {
		return false, runRecord{}, fmt.Errorf("unexpected claim result for run %s: %v", key, res)
	}
	claimedFlag, _ := fields[0].(int64)
	status, _ := fields[1].(string)
	jn, _ := fields[2].(string)
	attempts, _ := fields[3].(int64)

	return claimedFlag == 1, runRecord{Status: status, JobName: jn, Attempts: int(attempts)}, nil
}

func (c *Client) releaseRun(ctx context.Context, key string) {
	if err := c.redis.Del(ctx, runKey(key)).Err(); err != nil {
		slog.Warn("failed to release run claim after job creation error", "key", key, "error", err)
	}
}

func runKey(key string) string {
	return "run:" + key
}

// outputExists reports whether a prior run already wrote output under gcsPath.
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
