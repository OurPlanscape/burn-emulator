package dispatch

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"sort"
	"sync"
	"time"

	"github.com/golang/geo/s2"
	"github.com/wroge/wgs84"
	storage "google.golang.org/api/storage/v1"
	_ "modernc.org/sqlite"
)

// under <inputs>/<inputs_version>/, EPSG:5070
const (
	validVarLocsObject = "varlocs/valid_varlocs.gpkg" // varlocs with a trained model
	allVarLocsObject   = "varlocs/all_varlocs.gpkg"   // every varloc; untrained ones run PT
)

var (
	ErrInvalidTreatmentArea = errors.New("invalid treatment_area")
	ErrOutsideVarLocs       = errors.New("treatment_area does not intersect any varloc")
	ErrUnknownVarLoc        = errors.New("varloc not in all_varlocs.gpkg")
	ErrOutsideVarLoc        = errors.New("treatment_area does not intersect varloc")
)

// cells the treatment area is split into to score overlaps
const overlapCells = 512

// EPSG:5070, NAD83 / Conus Albers
var conusAlbers = wgs84.NAD83().AlbersEqualAreaConic(-96, 23, 29.5, 45.5, 0, 0)

type varLocShape struct {
	name    string
	polygon *s2.Polygon
	bound   s2.Rect
}

type cachedVarLocShapes struct {
	shapes     []varLocShape
	generation int64
	checked    time.Time
}

// caches each gpkg per inputs_version; re-downloads on a new GCS generation
type varLocsResolver struct {
	storage *storage.Service
	bucket  string
	prefix  string

	mu    sync.Mutex
	cache map[string]*cachedVarLocShapes
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
		cache:   map[string]*cachedVarLocShapes{},
	}, nil
}

func (r *varLocsResolver) shapes(ctx context.Context, inputsVersion, object string) ([]varLocShape, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	name := joinPath(r.prefix, inputsVersion, object)
	c := r.cache[name]
	if c != nil && time.Since(c.checked) < versionCacheTTL {
		return c.shapes, nil
	}

	obj, err := r.storage.Objects.Get(r.bucket, name).Context(ctx).Do()
	if err != nil {
		return nil, fmt.Errorf("reading varlocs gs://%s/%s: %w", r.bucket, name, err)
	}
	if c != nil && c.generation == obj.Generation {
		c.checked = time.Now()
		return c.shapes, nil
	}

	shapes, err := r.download(ctx, name, obj.Generation)
	if err != nil {
		return nil, err
	}
	r.cache[name] = &cachedVarLocShapes{shapes: shapes, generation: obj.Generation, checked: time.Now()}
	slog.Info("loaded varlocs", "object", name, "generation", obj.Generation, "varlocs", len(shapes))
	return shapes, nil
}

// sqlite reads the gpkg from a temp file
func (r *varLocsResolver) download(ctx context.Context, name string, generation int64) ([]varLocShape, error) {
	resp, err := r.storage.Objects.Get(r.bucket, name).Generation(generation).Context(ctx).Download()
	if err != nil {
		return nil, fmt.Errorf("downloading varlocs gs://%s/%s: %w", r.bucket, name, err)
	}
	defer resp.Body.Close()

	f, err := os.CreateTemp("", "varlocs-*.gpkg")
	if err != nil {
		return nil, err
	}
	defer os.Remove(f.Name())
	if _, err := io.Copy(f, resp.Body); err != nil {
		f.Close()
		return nil, fmt.Errorf("downloading varlocs gs://%s/%s: %w", r.bucket, name, err)
	}
	if err := f.Close(); err != nil {
		return nil, err
	}

	shapes, err := loadVarLocShapes(f.Name())
	if err != nil {
		return nil, fmt.Errorf("loading varlocs gs://%s/%s: %w", r.bucket, name, err)
	}
	return shapes, nil
}

func loadVarLocShapes(path string) ([]varLocShape, error) {
	db, err := sql.Open("sqlite", "file:"+path+"?mode=ro")
	if err != nil {
		return nil, err
	}
	defer db.Close()

	var table, column string
	var srsID int
	err = db.QueryRow(`SELECT table_name, column_name, srs_id FROM gpkg_geometry_columns LIMIT 1`).
		Scan(&table, &column, &srsID)
	if err != nil {
		return nil, fmt.Errorf("reading gpkg_geometry_columns: %w", err)
	}
	if srsID != 5070 {
		return nil, fmt.Errorf("layer %s is EPSG:%d, expected EPSG:5070", table, srsID)
	}

	rows, err := db.Query(fmt.Sprintf(`SELECT varloc, %q FROM %q`, column, table))
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	toLonLat := wgs84.Transform(conusAlbers, wgs84.LonLat())
	var shapes []varLocShape
	for rows.Next() {
		var name string
		var blob []byte
		if err := rows.Scan(&name, &blob); err != nil {
			return nil, err
		}
		wkb, err := gpkgWKB(blob)
		if err != nil {
			return nil, fmt.Errorf("varloc %s: %w", name, err)
		}
		polys, err := parseWKBPolygons(wkb)
		if err != nil {
			return nil, fmt.Errorf("varloc %s: %w", name, err)
		}
		for _, rings := range polys {
			for _, ring := range rings {
				for i, c := range ring {
					lon, lat, _ := toLonLat(c[0], c[1], 0)
					ring[i] = [2]float64{lon, lat}
				}
			}
		}
		p := s2Polygon(polys)
		shapes = append(shapes, varLocShape{name: name, polygon: p, bound: p.RectBound()})
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	if len(shapes) == 0 {
		return nil, errors.New("no varloc features")
	}
	return shapes, nil
}

// varloc: the requested one if it intersects the area in all_varlocs.gpkg, else the largest
// overlap there. DL needs the varloc in valid_varlocs.gpkg and a <varloc>/current; otherwise PT.
func (c *Client) selectVarLoc(ctx context.Context, inputsVersion, geojson, requested, backend string) (varLoc, modelVersion, selected string, err error) {
	polys, err := treatmentAreaLonLat(geojson)
	if err != nil {
		return "", "", "", fmt.Errorf("%w: %v", ErrInvalidTreatmentArea, err)
	}
	area := s2Polygon(polys)

	all, err := c.varLocs.shapes(ctx, inputsVersion, allVarLocsObject)
	if err != nil {
		return "", "", "", err
	}
	if requested != "" {
		known, hit := intersects(all, requested, area)
		if !known {
			return "", "", "", ErrUnknownVarLoc
		}
		if !hit {
			return "", "", "", ErrOutsideVarLoc
		}
		varLoc = requested
	} else {
		varLoc = largestOverlap(all, area)
		if varLoc == "" {
			return "", "", "", ErrOutsideVarLocs
		}
	}
	if backend == BackendPT {
		return varLoc, ptModelVersion, BackendPT, nil
	}

	valid, err := c.varLocs.shapes(ctx, inputsVersion, validVarLocsObject)
	if err != nil {
		return "", "", "", err
	}
	if known, _ := intersects(valid, varLoc, nil); !known {
		slog.Warn("varloc has no trained model; falling back to PT", "varloc", varLoc)
		return varLoc, ptModelVersion, BackendPT, nil
	}
	modelVersion, err = c.modelVersions.resolve(ctx, varLoc)
	if isStatusCode(err, 404) {
		slog.Warn("varloc has no released model; falling back to PT", "varloc", varLoc)
		return varLoc, ptModelVersion, BackendPT, nil
	}
	if err != nil {
		return "", "", "", fmt.Errorf("resolving model version for %s: %w", varLoc, err)
	}
	return varLoc, modelVersion, BackendDL, nil
}

// known: name is in shapes; hit: one of its shapes intersects area (area nil: not checked)
func intersects(shapes []varLocShape, name string, area *s2.Polygon) (known, hit bool) {
	for _, s := range shapes {
		if s.name != name {
			continue
		}
		known = true
		if area != nil && s.bound.Intersects(area.RectBound()) && s.polygon.Intersects(area) {
			return true, true
		}
	}
	return known, false
}

// scores covering cells whose center both contain; ties go to the first name
func largestOverlap(shapes []varLocShape, area *s2.Polygon) string {
	bound := area.RectBound()
	scores := map[string]float64{}
	for _, s := range shapes {
		if s.bound.Intersects(bound) && s.polygon.Intersects(area) {
			scores[s.name] = scores[s.name]
		}
	}
	if len(scores) == 0 {
		return ""
	}
	if len(scores) > 1 {
		coverer := &s2.RegionCoverer{MaxLevel: 30, MaxCells: overlapCells}
		for _, id := range coverer.Covering(area) {
			center := id.Point()
			if !area.ContainsPoint(center) {
				continue
			}
			w := s2.CellFromCellID(id).ApproxArea()
			for _, s := range shapes {
				if _, ok := scores[s.name]; ok && s.polygon.ContainsPoint(center) {
					scores[s.name] += w
				}
			}
		}
	}
	names := make([]string, 0, len(scores))
	for n := range scores {
		names = append(names, n)
	}
	sort.Strings(names)
	best := names[0]
	for _, n := range names[1:] {
		if scores[n] > scores[best] {
			best = n
		}
	}
	return best
}

// loops are normalized; nested rings become holes
func s2Polygon(polys [][][][2]float64) *s2.Polygon {
	var loops []*s2.Loop
	for _, rings := range polys {
		for _, ring := range rings {
			pts := make([]s2.Point, 0, len(ring))
			for _, c := range ring {
				p := s2.PointFromLatLng(s2.LatLngFromDegrees(c[1], c[0]))
				if len(pts) > 0 && pts[len(pts)-1] == p {
					continue
				}
				pts = append(pts, p)
			}
			if len(pts) > 1 && pts[0] == pts[len(pts)-1] {
				pts = pts[:len(pts)-1]
			}
			if len(pts) < 3 {
				continue
			}
			l := s2.LoopFromPoints(pts)
			l.Normalize()
			loops = append(loops, l)
		}
	}
	return s2.PolygonFromLoops(loops)
}
