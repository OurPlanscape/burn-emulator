import argparse
import sys

import geopandas as gpd
import pyogrio

from burn_emulator.constants import OUTDIR, VARLOCS_GPKG, Path

VARLOCS_TXT = OUTDIR / "varlocs.txt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path, default=VARLOCS_GPKG)
    parser.add_argument("-v", "--varlocs", type=Path, default=VARLOCS_TXT)
    parser.add_argument("-o", "--output", type=Path, default=OUTDIR / "valid_varlocs.gpkg")
    args = parser.parse_args()

    varlocs = [line.strip() for line in args.varlocs.read_text().splitlines() if line.strip()]
    layer = pyogrio.list_layers(args.input)[0][0]
    gdf = gpd.read_file(args.input, layer=layer)

    out = gdf[gdf["varloc"].isin(varlocs)]
    missing = sorted(set(varlocs) - set(out["varloc"]))
    if missing:
        print(f"warning: not found in {args.input}: {missing}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(args.output, layer=layer, driver="GPKG")
    print(f"wrote {len(out)} features ({out['varloc'].nunique()} varlocs) to {args.output}")


if __name__ == "__main__":
    main()
