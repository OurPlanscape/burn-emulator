package dispatch

import (
	"context"
	"fmt"
	"log/slog"
	"strconv"
	"strings"
	"time"

	storage "google.golang.org/api/storage/v1"
)

const (
	pruneMarker    = "_prune/last" // its update time is the last prune; rewritten with a generation precondition
	failedPrefix   = "_failed/"
	pruneEvery     = 1 * time.Hour
	claimRetention = 24 * time.Hour // failed and stale claims older than this move to _failed/
	pruneTimeout   = 30 * time.Second
	pruneMaxClaims = 500
)

// archives old claims at most once per pruneEvery across all instances; errors are logged.
// HandleReport runs it in its own goroutine.
func (c *Client) maybePrune(ctx context.Context) {
	bucket, err := c.outputBucketName()
	if err != nil {
		return
	}
	ctx, cancel := context.WithTimeout(ctx, pruneTimeout)
	defer cancel()

	var gen int64
	marker, err := c.storage.Objects.Get(bucket, pruneMarker).Context(ctx).Do()
	switch {
	case err == nil:
		last, perr := time.Parse(time.RFC3339, marker.Updated)
		if perr == nil && time.Since(last) < pruneEvery {
			return
		}
		gen = marker.Generation
	case !isStatusCode(err, 404):
		slog.Warn("failed to read prune marker", "error", err)
		return
	}

	_, err = c.storage.Objects.Insert(bucket, &storage.Object{Name: pruneMarker}).
		Media(strings.NewReader("")).
		IfGenerationMatch(gen).
		Context(ctx).
		Do()
	if err != nil {
		if !isStatusCode(err, 412) {
			slog.Warn("failed to write prune marker", "error", err)
		}
		return
	}

	archived, err := c.archiveClaims(ctx, bucket)
	if err != nil {
		slog.Warn("claim prune stopped early", "archived", archived, "error", err)
		return
	}
	slog.Info("claim prune done", "archived", archived)
}

// moves failed and stale claims older than claimRetention to _failed/<job_id>
func (c *Client) archiveClaims(ctx context.Context, bucket string) (int, error) {
	archived := 0
	pageToken := ""
	for {
		resp, err := c.storage.Objects.List(bucket).Prefix(claimPrefix).PageToken(pageToken).Context(ctx).Do()
		if err != nil {
			return archived, fmt.Errorf("listing claims: %w", err)
		}
		for _, obj := range resp.Items {
			if archived >= pruneMaxClaims {
				return archived, nil
			}
			rec := parseClaim(obj)
			if time.Since(rec.UpdatedAt) < claimRetention {
				continue
			}
			if err := c.archiveClaim(ctx, bucket, obj, rec); err != nil {
				return archived, err
			}
			archived++
		}
		if resp.NextPageToken == "" {
			return archived, nil
		}
		pageToken = resp.NextPageToken
	}
}

func (c *Client) archiveClaim(ctx context.Context, bucket string, obj *storage.Object, rec claimRecord) error {
	key := strings.TrimPrefix(obj.Name, claimPrefix)
	meta := map[string]string{
		"status":      statusFailed,
		"job_name":    rec.JobName,
		"attempts":    strconv.Itoa(rec.Attempts),
		"updated_at":  strconv.FormatInt(rec.UpdatedAt.Unix(), 10),
		"archived_at": strconv.FormatInt(time.Now().Unix(), 10),
		"error":       rec.Error,
	}
	if rec.Status != statusFailed {
		meta["last_status"] = rec.Status
		meta["error"] = "stale: no report"
	}
	_, err := c.storage.Objects.Insert(bucket, &storage.Object{Name: failedPrefix + key, Metadata: meta}).
		Media(strings.NewReader("")).
		Context(ctx).
		Do()
	if err != nil {
		return fmt.Errorf("archiving claim %s: %w", key, err)
	}

	err = c.storage.Objects.Delete(bucket, obj.Name).IfGenerationMatch(obj.Generation).Context(ctx).Do()
	if err != nil && !isStatusCode(err, 404) && !isStatusCode(err, 412) {
		return fmt.Errorf("deleting archived claim %s: %w", key, err)
	}
	slog.Info("claim archived", "key", key, "status", rec.Status)
	return nil
}

// the claim archived by archiveClaims, if any
func (c *Client) readArchivedClaim(ctx context.Context, key string) (claimRecord, bool, error) {
	bucket, err := c.outputBucketName()
	if err != nil {
		return claimRecord{}, false, err
	}
	obj, err := c.storage.Objects.Get(bucket, failedPrefix+key).Context(ctx).Do()
	if err != nil {
		if isStatusCode(err, 404) {
			return claimRecord{}, false, nil
		}
		return claimRecord{}, false, fmt.Errorf("reading archived claim %s: %w", key, err)
	}
	return parseClaim(obj), true, nil
}
