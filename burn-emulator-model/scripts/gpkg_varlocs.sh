#!/bin/bash
set -euo pipefail

# varlocs in constants.VARLOCS_GPKG, sorted; run from burn-emulator-model/ with the venv active

python -c "import geopandas as gpd; from burn_emulator.constants import VARLOCS_GPKG; print(*sorted(gpd.read_file(VARLOCS_GPKG, ignore_geometry=True)['varloc'].unique()), sep='\n')"
