#!/bin/bash
set -euo pipefail

# every varloc in the varlocs gpkg (constants.VARLOCS_GPKG), one per line, sorted;
# run from burn-emulator-model/ with the model venv active

python -c "import geopandas as gpd; from burn_emulator.constants import VARLOCS_GPKG; print(*sorted(gpd.read_file(VARLOCS_GPKG, ignore_geometry=True)['varloc'].unique()), sep='\n')"
