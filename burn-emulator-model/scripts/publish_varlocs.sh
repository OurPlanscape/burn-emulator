#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/load_env.sh"

# uploads both gpkgs over the varlocs layer of <inputs_root>/current's published version; the local
# inputs dir is left alone and current is not repointed
inputs_root="${1:-}"
valid_gpkg="${2:-}"
all_gpkg="${3:-}"
inputs_uri="${4:-${BURN_EMULATOR_INPUTS_URI:-}}"

if [[ -z "$inputs_root" || -z "$valid_gpkg" || -z "$all_gpkg" ]]; then
    echo "usage: $0 <inputs_root> <valid_varlocs.gpkg> <all_varlocs.gpkg> [inputs_uri]" >&2
    exit 2
fi
if [[ -z "$inputs_uri" ]]; then
    echo "error: pass inputs_uri as arg 4, or run make env-vars" >&2
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
if [[ ! -f "${inputs_root%/}/current" ]]; then
    echo "error: ${inputs_root%/}/current not found, run 'make inputs-current'" >&2
    exit 1
fi
data_version="$(<"${inputs_root%/}/current")"
dest="${inputs_uri%/}/${data_version}/varlocs"
if ! gcloud storage ls "${dest}/" >/dev/null 2>&1; then
    echo "error: ${dest}/ does not exist; run 'make inputs-release' first" >&2
    exit 1
fi

echo "data_version  ${data_version}"
echo "to            ${dest}/"
echo

for f in "$valid_gpkg" "$all_gpkg"; do
    name="$(basename "$f")"
    if gcloud storage cat "${dest}/${name}" 2>/dev/null | cmp -s - "$f"; then
        echo "unchanged: ${name}"
    else
        gcloud storage cp "$f" "${dest}/${name}"
        echo "updated: ${name}"
    fi
done
