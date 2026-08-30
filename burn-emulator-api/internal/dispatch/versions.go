package dispatch

import (
	"context"
	"fmt"
	"io"
	"regexp"
	"strings"
	"sync"
	"time"

	storage "google.golang.org/api/storage/v1"
)

// how long a resolved varloc -> version mapping is reused before the pointer
// is re-read. A model bump is picked up within this window.
const versionCacheTTL = 60 * time.Second

// versions must be usable as a single GCS path segment.
var validVersion = regexp.MustCompile(`^[A-Za-z0-9._-]{1,128}$`)

type cachedVersion struct {
	version string
	at      time.Time
}

// resolve and cache the current model version for a varloc, read from the
// gs://<models>/<varloc>/current pointer object (a one-line text file).
type versionResolver struct {
	storage *storage.Service
	bucket  string
	prefix  string // registry prefix within the bucket, may be ""

	mu    sync.Mutex
	cache map[string]cachedVersion
}

func newVersionResolver(storageSvc *storage.Service, modelsURI string) (*versionResolver, error) {
	bucket, prefix, err := parseGSRoot(modelsURI)
	if err != nil {
		return nil, fmt.Errorf("models URI: %w", err)
	}
	return &versionResolver{
		storage: storageSvc,
		bucket:  bucket,
		prefix:  prefix,
		cache:   map[string]cachedVersion{},
	}, nil
}

func (r *versionResolver) resolve(ctx context.Context, varloc string) (string, error) {
	r.mu.Lock()
	if c, ok := r.cache[varloc]; ok && time.Since(c.at) < versionCacheTTL {
		r.mu.Unlock()
		return c.version, nil
	}
	r.mu.Unlock()

	name := joinPath(r.prefix, varloc, "current")
	resp, err := r.storage.Objects.Get(r.bucket, name).Context(ctx).Download()
	if err != nil {
		return "", fmt.Errorf("reading version pointer gs://%s/%s: %w", r.bucket, name, err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 256))
	if err != nil {
		return "", fmt.Errorf("reading version pointer gs://%s/%s: %w", r.bucket, name, err)
	}
	version := strings.TrimSpace(string(body))
	if !validVersion.MatchString(version) {
		return "", fmt.Errorf("version pointer gs://%s/%s holds invalid version %q", r.bucket, name, version)
	}

	r.mu.Lock()
	r.cache[varloc] = cachedVersion{version: version, at: time.Now()}
	r.mu.Unlock()
	return version, nil
}

// split gs://<bucket>[/<prefix>] into bucket and (possibly empty) prefix.
func parseGSRoot(uri string) (bucket, prefix string, err error) {
	const scheme = "gs://"
	if !strings.HasPrefix(uri, scheme) {
		return "", "", fmt.Errorf("%q is not a gs:// path", uri)
	}
	bucket, prefix, _ = strings.Cut(strings.TrimPrefix(uri, scheme), "/")
	if bucket == "" {
		return "", "", fmt.Errorf("%q has no bucket name", uri)
	}
	return bucket, strings.Trim(prefix, "/"), nil
}

// join non-empty path segments with "/".
func joinPath(parts ...string) string {
	var kept []string
	for _, p := range parts {
		if p != "" {
			kept = append(kept, p)
		}
	}
	return strings.Join(kept, "/")
}
