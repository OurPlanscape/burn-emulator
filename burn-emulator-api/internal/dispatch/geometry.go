package dispatch

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"

	"github.com/wroge/wgs84"
)

var errUnsupportedCRS = errors.New("unsupported treatment_area crs")

// EPSG:4326, urn:ogc:def:crs:EPSG::4326, ...
var epsgCode = regexp.MustCompile(`EPSG:+(\d+)$`)

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

// lon/lat polygons; no crs means EPSG:4326
func treatmentAreaLonLat(s string) ([][][][2]float64, error) {
	var g geoJSON
	if err := json.Unmarshal([]byte(s), &g); err != nil {
		return nil, err
	}

	var toLonLat func(x, y float64) (float64, float64)
	name := ""
	if g.CRS != nil {
		name = g.CRS.Properties.Name
	}
	code := 4326
	if m := epsgCode.FindStringSubmatch(name); m != nil {
		code, _ = strconv.Atoi(m[1])
	} else if name != "" && !strings.HasSuffix(name, "CRS84") {
		return nil, fmt.Errorf("%w: %q", errUnsupportedCRS, name)
	}
	switch code {
	case 4326, 4269:
	case 5070:
		t := wgs84.Transform(conusAlbers, wgs84.LonLat())
		toLonLat = func(x, y float64) (float64, float64) { lon, lat, _ := t(x, y, 0); return lon, lat }
	default:
		crs := wgs84.EPSG().Code(code)
		if crs == nil {
			return nil, fmt.Errorf("%w: EPSG:%d", errUnsupportedCRS, code)
		}
		t := wgs84.Transform(crs, wgs84.LonLat())
		toLonLat = func(x, y float64) (float64, float64) { lon, lat, _ := t(x, y, 0); return lon, lat }
	}

	var polys [][][][2]float64
	if err := collectPolygons(g, &polys); err != nil {
		return nil, err
	}
	if len(polys) == 0 {
		return nil, errors.New("no Polygon or MultiPolygon geometry")
	}
	if toLonLat != nil {
		for _, rings := range polys {
			for _, ring := range rings {
				for i, c := range ring {
					lon, lat := toLonLat(c[0], c[1])
					ring[i] = [2]float64{lon, lat}
				}
			}
		}
	}
	return polys, nil
}

func collectPolygons(g geoJSON, out *[][][][2]float64) error {
	switch g.Type {
	case "FeatureCollection":
		for _, f := range g.Features {
			if err := collectPolygons(f, out); err != nil {
				return err
			}
		}
	case "Feature":
		if g.Geometry != nil {
			return collectPolygons(*g.Geometry, out)
		}
	case "GeometryCollection":
		for _, sub := range g.Geometries {
			if err := collectPolygons(sub, out); err != nil {
				return err
			}
		}
	case "Polygon":
		var rings [][][]float64
		if err := json.Unmarshal(g.Coordinates, &rings); err != nil {
			return fmt.Errorf("Polygon coordinates: %w", err)
		}
		p, err := toRings(rings)
		if err != nil {
			return err
		}
		*out = append(*out, p)
	case "MultiPolygon":
		var polys [][][][]float64
		if err := json.Unmarshal(g.Coordinates, &polys); err != nil {
			return fmt.Errorf("MultiPolygon coordinates: %w", err)
		}
		for _, rings := range polys {
			p, err := toRings(rings)
			if err != nil {
				return err
			}
			*out = append(*out, p)
		}
	}
	return nil
}

func toRings(rings [][][]float64) ([][][2]float64, error) {
	out := make([][][2]float64, len(rings))
	for i, ring := range rings {
		out[i] = make([][2]float64, len(ring))
		for j, c := range ring {
			if len(c) < 2 || math.IsNaN(c[0]) || math.IsNaN(c[1]) {
				return nil, errors.New("coordinate needs x and y")
			}
			out[i][j] = [2]float64{c[0], c[1]}
		}
	}
	return out, nil
}

// strip the GeoPackage geometry header (magic, version, flags, srs_id, envelope)
func gpkgWKB(b []byte) ([]byte, error) {
	if len(b) < 8 || b[0] != 'G' || b[1] != 'P' {
		return nil, errors.New("not a GeoPackage geometry")
	}
	envSizes := [...]int{0, 32, 48, 48, 64}
	env := int(b[3]>>1) & 0x7
	if env >= len(envSizes) {
		return nil, fmt.Errorf("bad envelope indicator %d", env)
	}
	n := 8 + envSizes[env]
	if len(b) < n {
		return nil, errors.New("truncated GeoPackage geometry")
	}
	return b[n:], nil
}

type wkbReader struct {
	b   []byte
	pos int
}

func (r *wkbReader) byteOrder() (binary.ByteOrder, error) {
	if r.pos >= len(r.b) {
		return nil, errors.New("truncated wkb")
	}
	o := r.b[r.pos]
	r.pos++
	if o == 0 {
		return binary.BigEndian, nil
	}
	return binary.LittleEndian, nil
}

func (r *wkbReader) uint32(o binary.ByteOrder) (uint32, error) {
	if r.pos+4 > len(r.b) {
		return 0, errors.New("truncated wkb")
	}
	v := o.Uint32(r.b[r.pos:])
	r.pos += 4
	return v, nil
}

func (r *wkbReader) float64(o binary.ByteOrder) (float64, error) {
	if r.pos+8 > len(r.b) {
		return 0, errors.New("truncated wkb")
	}
	v := math.Float64frombits(o.Uint64(r.b[r.pos:]))
	r.pos += 8
	return v, nil
}

// Polygon / MultiPolygon WKB (ISO or EWKB Z/M flags) -> polygons of x/y rings
func parseWKBPolygons(b []byte) ([][][][2]float64, error) {
	r := &wkbReader{b: b}
	var out [][][][2]float64
	if err := r.geometry(&out); err != nil {
		return nil, err
	}
	return out, nil
}

func (r *wkbReader) geometry(out *[][][][2]float64) error {
	o, err := r.byteOrder()
	if err != nil {
		return err
	}
	t, err := r.uint32(o)
	if err != nil {
		return err
	}
	dims := 2
	if t&0x80000000 != 0 {
		dims++
	}
	if t&0x40000000 != 0 {
		dims++
	}
	t &^= 0xC0000000
	switch t / 1000 {
	case 1, 2:
		dims++
	case 3:
		dims += 2
	}

	switch t % 1000 {
	case 3:
		p, err := r.polygon(o, dims)
		if err != nil {
			return err
		}
		*out = append(*out, p)
	case 6:
		n, err := r.uint32(o)
		if err != nil {
			return err
		}
		for range n {
			if err := r.geometry(out); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("unsupported wkb geometry type %d", t)
	}
	return nil
}

func (r *wkbReader) polygon(o binary.ByteOrder, dims int) ([][][2]float64, error) {
	nRings, err := r.uint32(o)
	if err != nil {
		return nil, err
	}
	rings := make([][][2]float64, 0, nRings)
	for range nRings {
		nPts, err := r.uint32(o)
		if err != nil {
			return nil, err
		}
		if int(nPts)*dims*8 > len(r.b)-r.pos {
			return nil, errors.New("truncated wkb")
		}
		ring := make([][2]float64, nPts)
		for i := range ring {
			for d := range dims {
				v, err := r.float64(o)
				if err != nil {
					return nil, err
				}
				if d < 2 {
					ring[i][d] = v
				}
			}
		}
		rings = append(rings, ring)
	}
	return rings, nil
}
