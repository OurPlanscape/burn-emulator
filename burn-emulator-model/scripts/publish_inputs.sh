#!/usr/bin/env bash

set -euo pipefail

data_version_raw="${1:-}"
fuels_dir="${2:-}"
topo_dir="${3:-}"
valid_gpkg="${4:-}"
all_gpkg="${5:-}"
fbfm_map="${6:-}"
inputs_uri="${7:-${BURN_EMULATOR_INPUTS_URI:-}}"

if [[ -z "$data_version_raw" || -z "$fuels_dir" || -z "$topo_dir" || -z "$valid_gpkg" || -z "$all_gpkg" || -z "$fbfm_map" ]]; then
    echo "usage: $0 <data_version> <fuels_dir> <topo_dir> <valid_varlocs.gpkg> <all_varlocs.gpkg> <fbfm_map> [inputs_uri]" >&2
    exit 2
fi
if [[ -z "$inputs_uri" ]]; then
    echo "error: pass inputs_uri as arg 7, or set BURN_EMULATOR_INPUTS_URI" >&2
    exit 2
fi

# data_version: west fuels date, YYYYMMDD
if [[ "$data_version_raw" =~ ^[0-9]{8}$ ]]; then
    data_version="$data_version_raw"
else
    echo "error: data_version '$data_version_raw' must be YYYYMMDD" >&2
    exit 1
fi

for dir in "$fuels_dir" "$topo_dir"; do
    if [[ ! -d "$dir" ]]; then
        echo "error: $dir is not a directory" >&2
        exit 1
    fi
done
if ! compgen -G "$topo_dir/*.tif" >/dev/null; then
    echo "error: $topo_dir has no *.tif layers" >&2
    exit 1
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
if [[ ! -f "$fbfm_map" || "$(basename "$fbfm_map")" != fbfm_behavior_adjectives.csv ]]; then
    echo "error: $fbfm_map is not a fbfm_behavior_adjectives.csv file" >&2
    exit 1
fi

# fuels_dir/{baseline,legalmax}/<layer>.tif
for t in baseline legalmax; do
    for l in cbd cbh cc fbfm th; do
        [[ -f "$fuels_dir/$t/$l.tif" ]] || { echo "error: $fuels_dir/$t/$l.tif not found" >&2; exit 1; }
    done
done

base="${inputs_uri%/}/${data_version}"

# skips a published layer unless FORCE=1
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

publish_layer baseline "$fuels_dir"/baseline/*.tif
publish_layer legalmax "$fuels_dir"/legalmax/*.tif
publish_layer topo "$topo_dir"/*.tif
publish_layer varlocs "$valid_gpkg" "$all_gpkg"
publish_layer fbfm "$fbfm_map"

# repoints current after every layer is up
printf '%s' "$data_version" | gcloud storage cp - "${inputs_uri%/}/current"
echo "current now points to ${data_version}"
