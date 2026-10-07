#!/usr/bin/env bash

set -euo pipefail

valid_gpkg="${1:-}"
all_gpkg="${2:-}"
data_version_raw="${3:-}"
inputs_uri="${4:-${BURN_EMULATOR_INPUTS_URI:-}}"

if [[ -z "$valid_gpkg" || -z "$all_gpkg" ]]; then
    echo "usage: $0 <valid_varlocs.gpkg> <all_varlocs.gpkg> [data_version (default: current)] [inputs_uri]" >&2
    exit 2
fi
if [[ -z "$inputs_uri" ]]; then
    echo "error: pass inputs_uri as arg 4, or set BURN_EMULATOR_INPUTS_URI" >&2
    exit 2
fi
# names the api reads
for pair in "$valid_gpkg:valid_varlocs.gpkg" "$all_gpkg:all_varlocs.gpkg"; do
    f="${pair%:*}"
    name="${pair##*:}"
    if [[ ! -f "$f" || "$(basename "$f")" != "$name" ]]; then
        echo "error: $f is not a $name file" >&2
        exit 1
    fi
done

if [[ -z "$data_version_raw" ]]; then
    data_version="$(gcloud storage cat "${inputs_uri%/}/current")"
elif [[ "$data_version_raw" =~ ^[0-9]{8}$ ]]; then
    data_version="$data_version_raw"
else
    echo "error: data_version '$data_version_raw' must be YYYYMMDD" >&2
    exit 1
fi

# replaces only the varlocs layer; current is not repointed
dest="${inputs_uri%/}/${data_version}/varlocs"
if ! gcloud storage ls "${dest}/" >/dev/null 2>&1; then
    echo "error: ${dest}/ does not exist; publish the data_version with publish_inputs.sh first" >&2
    exit 1
fi

echo "data_version  ${data_version}"
echo "to            ${dest}/"
echo

for f in "$valid_gpkg" "$all_gpkg"; do
    gpkg_dest="${dest}/$(basename "$f")"
    if gcloud storage cat "$gpkg_dest" 2>/dev/null | cmp -s - "$f"; then
        echo "unchanged: $(basename "$f")"
    else
        gcloud storage cp "$f" "$gpkg_dest"
        echo "updated: $(basename "$f")"
    fi
done
