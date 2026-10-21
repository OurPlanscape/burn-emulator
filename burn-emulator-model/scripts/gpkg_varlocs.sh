#!/bin/bash
set -euo pipefail

# varlocs in data/inputs/<inputs_version>/varlocs/all_varlocs.gpkg (default: current.yaml's), sorted;
# run from burn-emulator-model/ with the venv active
version="${1:-$("$(dirname "${BASH_SOURCE[0]}")"/data_version.sh inputs_version)}"

python -c "import sys, geopandas as gpd; from burn_emulator.config import varlocs_gpkg; print(*sorted(gpd.read_file(varlocs_gpkg(sys.argv[1]), ignore_geometry=True)['varloc'].unique()), sep='\n')" "$version"
