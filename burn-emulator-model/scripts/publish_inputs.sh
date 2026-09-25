#!/usr/bin/env bash

set -euo pipefail

data_version_raw="${1:-}"
fuels_dir="${2:-}"
topo_dir="${3:-}"
inputs_uri="${4:-${BURN_EMULATOR_INPUTS_URI:-}}"

if [[ -z "$data_version_raw" || -z "$fuels_dir" || -z "$topo_dir" ]]; then
    echo "usage: $0 <data_version> <fuels_dir> <topo_dir> [inputs_uri]" >&2
    exit 2
fi
if [[ -z "$inputs_uri" ]]; then
    echo "error: pass inputs_uri as arg 4, or set BURN_EMULATOR_INPUTS_URI" >&2
    exit 2
fi

# data_version as DDMonYYYY (28Aug2026) or YYYYMMDD, normalised to YYYYMMDD
if [[ "$data_version_raw" =~ ^[0-9]{8}$ ]]; then
    data_version="$data_version_raw"
elif [[ "$data_version_raw" =~ ^([0-9]{1,2})([A-Za-z]{3})([0-9]{4})$ ]]; then
    data_version="$(date -u -d "${BASH_REMATCH[1]} ${BASH_REMATCH[2]} ${BASH_REMATCH[3]}" +%Y%m%d)"
else
    echo "error: data_version '$data_version_raw' must be DDMonYYYY or YYYYMMDD" >&2
    exit 1
fi

for dir in "$fuels_dir" "$topo_dir"; do
    if [[ ! -d "$dir" ]]; then
        echo "error: $dir is not a directory" >&2
        exit 1
    fi
    if ! compgen -G "$dir/*.tif" >/dev/null; then
        echo "error: $dir has no *.tif layers" >&2
        exit 1
    fi
done

# fuels_dir holds both baseline and legalmax tifs together, split by filename
baseline_files=()
legalmax_files=()
unmatched_files=()
for f in "$fuels_dir"/*.tif; do
    lower="$(basename "$f")"
    lower="${lower,,}"
    if [[ "$lower" == *baseline* ]]; then
        baseline_files+=("$f")
    elif [[ "$lower" == *legalmax* ]]; then
        legalmax_files+=("$f")
    else
        unmatched_files+=("$f")
    fi
done

if [[ ${#unmatched_files[@]} -gt 0 ]]; then
    echo "error: fuels filenames must contain 'baseline' or 'legalmax', got: ${unmatched_files[*]}" >&2
    exit 1
fi
if [[ ${#baseline_files[@]} -eq 0 ]]; then
    echo "error: no *baseline*.tif files found in $fuels_dir" >&2
    exit 1
fi
if [[ ${#legalmax_files[@]} -eq 0 ]]; then
    echo "error: no *legalmax*.tif files found in $fuels_dir" >&2
    exit 1
fi

base="${inputs_uri%/}/${data_version}"

# dedup: skip a layer that's already published (FORCE=1 to re-upload)
publish_layer () {
    local layer="$1"
    shift
    local dest="${base}/${layer}"

    echo "layer         ${layer}"
    echo "data_version  ${data_version}"
    echo "files         $#"
    echo "to            ${dest}/"
    echo

    local cp_flags=(--no-clobber)
    if [[ "${FORCE:-0}" == "1" ]]; then
        cp_flags=()
    elif gcloud storage ls "${dest}/" >/dev/null 2>&1; then
        echo "already published: ${dest}/ exists (FORCE=1 to re-upload)"
        echo
        return
    fi

    gcloud storage cp "${cp_flags[@]}" "$@" "${dest}/"
    echo
    echo "done: ${layer} published to ${dest}/"
    echo
}

publish_layer baseline "${baseline_files[@]}"
publish_layer legalmax "${legalmax_files[@]}"
publish_layer topo "$topo_dir"/*.tif

# only repoint once every layer of this data_version is up (set -e stops earlier on failure)
printf '%s' "$data_version" | gcloud storage cp - "${inputs_uri%/}/current"
echo "current now points to ${data_version}"
