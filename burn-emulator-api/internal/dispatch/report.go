package dispatch

import (
	"context"
	"fmt"
	"log/slog"
	"strconv"
	"strings"
)

const (
	reportCompleted = "completed" // runner REPORT_COMPLETED
	reportFailed    = "failed"    // runner REPORT_FAILED
)

// handles a runner report named by a Pub/Sub storage notification
func (c *Client) HandleReport(ctx context.Context, bucket, object string, generation int64) error {
	outBucket, err := c.outputBucketName()
	if err != nil {
		return err
	}
	runObject, ok := strings.CutPrefix(object, reportPrefix)
	if bucket != outBucket || !ok {
		slog.Warn("ignoring notification for unexpected object", "bucket", bucket, "object", object)
		return nil
	}
	// <inputs_version>/<varloc>/<model_version>/<hash>, see JobID.Path
	if strings.Count(runObject, "/") != 3 || strings.Contains(runObject, "//") {
		slog.Warn("ignoring malformed report", "object", object)
		return nil
	}

	report, err := c.storage.Objects.Get(bucket, object).IfGenerationMatch(generation).Context(ctx).Do()
	if err != nil {
		if isStatusCode(err, 404) || isStatusCode(err, 412) {
			return nil // already handled, or replaced by a newer report
		}
		return fmt.Errorf("reading report %s: %w", object, err)
	}

	status := report.Metadata["status"]
	claimGen, err := strconv.ParseInt(report.Metadata["claim_generation"], 10, 64)
	if err != nil || claimGen == 0 {
		slog.Warn("report has no claim generation; leaving claim to go stale", "object", object)
		status = ""
	}

	switch status {
	case reportCompleted:
		c.releaseRun(ctx, runObject, claimGen)
	case reportFailed:
		if c.ownsClaim(ctx, runObject, claimGen) {
			c.deleteOutput(ctx, strings.TrimSuffix(c.cfg.OutputBucket, "/")+"/"+runObject)
			if err := c.failRun(ctx, runObject, claimGen, report.Metadata["error"]); err != nil {
				return err
			}
		}
	case "":
	default:
		slog.Warn("report has unknown status", "object", object, "status", status)
	}

	slog.Info("run finished", "run", runObject, "status", status, "error", report.Metadata["error"])

	err = c.storage.Objects.Delete(bucket, object).IfGenerationMatch(generation).Context(ctx).Do()
	if err != nil && !isStatusCode(err, 404) && !isStatusCode(err, 412) {
		slog.Warn("failed to delete report", "object", object, "error", err)
	}
	return nil
}
