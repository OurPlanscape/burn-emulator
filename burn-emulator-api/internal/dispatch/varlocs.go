package dispatch

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"

	storage "google.golang.org/api/storage/v1"
)

// <inputs>/<data_version>/varlocs/varlocs.txt, written by publish_inputs.sh
const varLocsObject = "varlocs/varlocs.txt"

var ErrUnknownVarLoc = errors.New("varloc not in the published allow-list")

type cachedVarLocs struct {
	set map[string]bool
	at  time.Time
}

// read and cache the varloc allow-list published with each data_version
type varLocsResolver struct {
	storage *storage.Service
	bucket  string
	prefix  string

	mu    sync.Mutex
	cache map[string]cachedVarLocs
}

func newVarLocsResolver(storageSvc *storage.Service, uri string) (*varLocsResolver, error) {
	bucket, prefix, err := parseGSRoot(uri)
	if err != nil {
		return nil, fmt.Errorf("varlocs resolver root: %w", err)
	}
	return &varLocsResolver{
		storage: storageSvc,
		bucket:  bucket,
		prefix:  prefix,
		cache:   map[string]cachedVarLocs{},
	}, nil
}

func (r *varLocsResolver) contains(ctx context.Context, dataVersion, varLoc string) (bool, error) {
	r.mu.Lock()
	if c, ok := r.cache[dataVersion]; ok && time.Since(c.at) < versionCacheTTL {
		r.mu.Unlock()
		return c.set[varLoc], nil
	}
	r.mu.Unlock()

	name := joinPath(r.prefix, dataVersion, varLocsObject)
	resp, err := r.storage.Objects.Get(r.bucket, name).Context(ctx).Download()
	if err != nil {
		return false, fmt.Errorf("reading varlocs gs://%s/%s: %w", r.bucket, name, err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return false, fmt.Errorf("reading varlocs gs://%s/%s: %w", r.bucket, name, err)
	}

	set := map[string]bool{}
	for _, line := range strings.Split(string(body), "\n") {
		if line = strings.TrimSpace(line); line != "" {
			set[line] = true
		}
	}
	if len(set) == 0 {
		return false, fmt.Errorf("varlocs gs://%s/%s lists no varlocs", r.bucket, name)
	}

	r.mu.Lock()
	r.cache[dataVersion] = cachedVarLocs{set: set, at: time.Now()}
	r.mu.Unlock()
	return set[varLoc], nil
}
