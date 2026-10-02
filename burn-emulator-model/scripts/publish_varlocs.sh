#!/usr/bin/env bash

set -euo pipefail

varlocs_txt="${1:-}"
varlocs_gpkg="${2:-}"
data_version_raw="${3:-}"
inputs_uri="${4:-${BURN_EMULATOR_INPUTS_URI:-}}"

if [[ -z "$varlocs_txt" || -z "$varlocs_gpkg" ]]; then
    echo "usage: $0 <varlocs_txt> <varlocs_gpkg> [data_version (default: current)] [inputs_uri]" >&2
    exit 2
fi
if [[ -z "$inputs_uri" ]]; then
    echo "error: pass inputs_uri as arg 4, or set BURN_EMULATOR_INPUTS_URI" >&2
    exit 2
fi
if ! grep -qvE '^[[:space:]]*$' "$varlocs_txt" 2>/dev/null; then
    echo "error: $varlocs_txt is missing or lists no varlocs" >&2
    exit 1
fi
if [[ ! -f "$varlocs_gpkg" || "$varlocs_gpkg" != *.gpkg ]]; then
    echo "error: $varlocs_gpkg is not a .gpkg file" >&2
    exit 1
fi

if [[ -z "$data_version_raw" ]]; then
    data_version="$(gcloud storage cat "${inputs_uri%/}/current")"
elif [[ "$data_version_raw" =~ ^[0-9]{8}$ ]]; then
    data_version="$data_version_raw"
else
    echo "error: data_version '$data_version_raw' must be YYYYMMDD" >&2
    exit 1
fi

# only replaces the varlocs layer of an already published data_version; never repoints current
dest="${inputs_uri%/}/${data_version}/varlocs"
txt_dest="${dest}/varlocs.txt"
gpkg_dest="${dest}/$(basename "$varlocs_gpkg")"
if ! gcloud storage ls "$txt_dest" >/dev/null 2>&1; then
    echo "error: $txt_dest does not exist; publish the data_version with publish_inputs.sh first" >&2
    exit 1
fi
mapfile -t published_gpkgs < <(gcloud storage ls "${dest}/*.gpkg" 2>/dev/null || true)
for g in "${published_gpkgs[@]}"; do
    if [[ "$g" != "$gpkg_dest" ]]; then
        echo "error: ${dest}/ has $(basename "$g"), not $(basename "$varlocs_gpkg"); rename the local gpkg to match" >&2
        exit 1
    fi
done

echo "data_version  ${data_version}"
echo "to            ${dest}/"
echo

if gcloud storage cat "$txt_dest" | cmp -s - "$varlocs_txt"; then
    echo "unchanged: $(basename "$txt_dest")"
else
    diff <(gcloud storage cat "$txt_dest" | grep -vE '^[[:space:]]*$' | LC_ALL=C sort -u) \
         <(grep -vE '^[[:space:]]*$' "$varlocs_txt" | LC_ALL=C sort -u) | grep -E '^[<>]' \
        | sed 's/^</  removed/; s/^>/  added  /' || true
    gcloud storage cp "$varlocs_txt" "$txt_dest"
    echo "updated: $(basename "$txt_dest") (api picks it up within its 60s cache)"
fi

if gcloud storage cat "$gpkg_dest" 2>/dev/null | cmp -s - "$varlocs_gpkg"; then
    echo "unchanged: $(basename "$gpkg_dest")"
else
    gcloud storage cp "$varlocs_gpkg" "$gpkg_dest"
    echo "updated: $(basename "$gpkg_dest")"
fi
