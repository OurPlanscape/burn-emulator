package dispatch

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"regexp"
	"strings"

	"github.com/twpayne/go-geos"
	"github.com/twpayne/go-proj/v10"
)

// equal-area CRS for varloc gpkgs and overlap scoring; runner TARGET_CRS
const (
	workingSRID = 5070
	workingCRS  = "EPSG:5070"
)

var (
	geosCtx          = geos.NewContext()
	errEmptyGeometry = errors.New("empty geometry")
	epsgCode         = regexp.MustCompile(`EPSG:+(\d+)$`) // EPSG:4326, urn:ogc:def:crs:EPSG::4326, ...
)

type geoJSON struct {
	Type        string          `json:"type"`
	Coordinates json.RawMessage `json:"coordinates"`
	Geometry    *geoJSON        `json:"geometry"`
	Features    []geoJSON       `json:"features"`
	Geometries  []geoJSON       `json:"geometries"`
	CRS         *struct {
		Properties struct {
			Name string `json:"name"`
		} `json:"properties"`
	} `json:"crs"`
}

// polygons reprojected to workingCRS; crs is required
func treatmentArea(s string) (*geos.Geom, error) {
	var g geoJSON
	if err := json.Unmarshal([]byte(s), &g); err != nil {
		return nil, err
	}
	if g.CRS == nil {
		return nil, errors.New("missing crs")
	}

	var src string
	name := g.CRS.Properties.Name
	if m := epsgCode.FindStringSubmatch(name); m != nil {
		src = "EPSG:" + m[1]
	} else if strings.HasSuffix(name, "CRS84") {
		src = "OGC:CRS84"
	} else {
		return nil, fmt.Errorf("unsupported crs %q", name)
	}

	var polys [][][][]float64
	if err := g.collect(&polys); err != nil {
		return nil, err
	}
	if len(polys) == 0 {
		return nil, errors.New("no Polygon or MultiPolygon geometry")
	}
	if err := reproject(src, polys); err != nil {
		return nil, err
	}

	// each part made valid then unioned, like the runner's union_all
	parts := make([]*geos.Geom, 0, len(polys))
	for _, rings := range polys {
		for _, ring := range rings {
			if len(ring) < 4 || !closed(ring) {
				return nil, errors.New("polygon rings need 4+ positions and must be closed")
			}
		}
		p, err := validPolygons(geosCtx.NewPolygon(rings))
		if errors.Is(err, errEmptyGeometry) {
			continue
		}
		if err != nil {
			return nil, err
		}
		parts = append(parts, p)
	}
	area := geosCtx.NewCollection(geos.TypeIDGeometryCollection, parts).UnaryUnion()
	if area.IsEmpty() {
		return nil, errEmptyGeometry
	}
	return area, nil
}

// area as a one-Feature FeatureCollection in workingCRS, crs member included
func workingGeoJSON(area *geos.Geom) string {
	return `{"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"` + workingCRS +
		`"}},"features":[{"type":"Feature","properties":{},"geometry":` + area.ToGeoJSON(0) + `}]}`
}

// FeatureCollection of Features, a Feature, or a bare geometry
func (g geoJSON) collect(out *[][][][]float64) error {
	switch g.Type {
	case "FeatureCollection":
		for _, f := range g.Features {
			if f.Type != "Feature" {
				return fmt.Errorf("FeatureCollection member type %q", f.Type)
			}
			if err := f.collect(out); err != nil {
				return err
			}
		}
		return nil
	case "Feature":
		if g.Geometry == nil {
			return nil
		}
		return g.Geometry.polygons(out)
	}
	return g.polygons(out)
}

func (g geoJSON) polygons(out *[][][][]float64) error {
	switch g.Type {
	case "GeometryCollection":
		for _, sub := range g.Geometries {
			if err := sub.polygons(out); err != nil {
				return err
			}
		}
	case "Polygon":
		var p [][][]float64
		if err := json.Unmarshal(g.Coordinates, &p); err != nil {
			return fmt.Errorf("Polygon coordinates: %w", err)
		}
		*out = append(*out, p)
	case "MultiPolygon":
		var p [][][][]float64
		if err := json.Unmarshal(g.Coordinates, &p); err != nil {
			return fmt.Errorf("MultiPolygon coordinates: %w", err)
		}
		*out = append(*out, p...)
	default:
		return fmt.Errorf("unsupported geometry type %q", g.Type)
	}
	return nil
}

// reprojects polys in place from src to workingCRS, x/y only; time unknown like pyproj
func reproject(src string, polys [][][][]float64) error {
	pj, err := proj.NewCRSToCRS(src, workingCRS, nil)
	if err != nil {
		return fmt.Errorf("%s: %w", src, err)
	}
	defer pj.Destroy()
	norm, err := pj.NormalizeForVisualization()
	if err != nil {
		return fmt.Errorf("%s: %w", src, err)
	}
	defer norm.Destroy()

	for _, rings := range polys {
		for i, ring := range rings {
			coords := make([]proj.Coord, len(ring))
			for j, c := range ring {
				if len(c) < 2 {
					return errors.New("position needs 2+ values")
				}
				coords[j] = proj.Coord{c[0], c[1], 0, math.Inf(1)}
			}
			if err := norm.ForwardArray(coords); err != nil {
				return fmt.Errorf("%s to %s: %w", src, workingCRS, err)
			}
			xy := make([][]float64, len(coords))
			for j, c := range coords {
				if math.IsInf(c[0], 0) || math.IsInf(c[1], 0) || math.IsNaN(c[0]) || math.IsNaN(c[1]) {
					return fmt.Errorf("%s to %s: position out of range", src, workingCRS)
				}
				xy[j] = []float64{c[0], c[1]}
			}
			rings[i] = xy
		}
	}
	return nil
}

func closed(ring [][]float64) bool {
	a, b := ring[0], ring[len(ring)-1]
	return a[0] == b[0] && a[1] == b[1]
}

// GeoPackage geometry blob (header + WKB) -> Polygon / MultiPolygon
func gpkgGeom(b []byte) (*geos.Geom, error) {
	if len(b) < 8 || b[0] != 'G' || b[1] != 'P' {
		return nil, errors.New("not a GeoPackage geometry")
	}
	env := (b[3] >> 1) & 7
	if env > 4 {
		return nil, fmt.Errorf("bad envelope indicator %d", env)
	}
	n := 8 + [...]int{0, 32, 48, 48, 64}[env]
	if len(b) < n {
		return nil, errors.New("truncated GeoPackage geometry")
	}
	g, err := geosCtx.NewGeomFromWKB(b[n:])
	if err != nil {
		return nil, err
	}
	return validPolygons(g)
}

// rejects non-polygonal or empty geometry; repairs invalid ones
func validPolygons(g *geos.Geom) (*geos.Geom, error) {
	switch g.TypeID() {
	case geos.TypeIDPolygon, geos.TypeIDMultiPolygon:
	default:
		return nil, fmt.Errorf("unsupported geometry type %s", g.Type())
	}
	if !g.IsValid() {
		g = g.MakeValidWithParams(geos.MakeValidStructure, geos.MakeValidDiscardCollapsed)
	}
	if g.IsEmpty() {
		return nil, errEmptyGeometry
	}
	return g, nil
}
