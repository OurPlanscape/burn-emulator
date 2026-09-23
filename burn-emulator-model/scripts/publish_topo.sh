#!/usr/bin/env bash

set -euo pipefail

topo_dir="${1:-}"
inputs_uri="${2:-${BURN_EMULATOR_INPUTS_URI:-}}"

if [[ -z "$topo_dir" ]]; then
    echo "usage: $0 <topo_dir> [inputs_uri]" >&2
    exit 2
fi
if [[ -z "$inputs_uri" ]]; then
    echo "error: pass inputs_uri as arg 2, or set BURN_EMULATOR_INPUTS_URI" >&2
    exit 2
fi

# topo_dir holds every topo tif as-is, no filename splitting
if [[ ! -d "$topo_dir" ]]; then
    echo "error: $topo_dir is not a directory" >&2
    exit 1
fi
if ! compgen -G "$topo_dir/*.tif" >/dev/null; then
    echo "error: $topo_dir has no *.tif layers" >&2
    exit 1
fi

# date = the DDMonYYYY stamp on the directory name, normalised to ISO YYYYMMDD
dir_name="$(basename "${topo_dir%/}")"
if [[ ! "$dir_name" =~ ([0-9]{1,2})([A-Za-z]{3})([0-9]{4}) ]]; then
    echo "error: no DDMonYYYY date in directory name '$dir_name'" >&2
    exit 1
fi
date_dir="$(date -u -d "${BASH_REMATCH[1]} ${BASH_REMATCH[2]} ${BASH_REMATCH[3]}" +%Y%m%d)"

base="${inputs_uri%/}/${date_dir}/topo"

echo "layer    topo"
echo "date     ${date_dir}"
echo "from     ${topo_dir}"
echo "to       ${base}/"
echo

# dedup: skip a layer directory that already exists (FORCE=1 to re-upload)
cp_flags=(--no-clobber)
if [[ "${FORCE:-0}" == "1" ]]; then
    cp_flags=()
elif gcloud storage ls "${base}/" >/dev/null 2>&1; then
    echo "already published: ${base}/ exists (FORCE=1 to re-upload)"
    printf '%s' "$date_dir" | gcloud storage cp - "${inputs_uri%/}/topo/current"
    echo "topo/current now points to ${date_dir}"
    exit 0
fi

gcloud storage cp "${cp_flags[@]}" "$topo_dir"/*.tif "${base}/"

echo
echo "done: topo layer published to ${base}/"
printf '%s' "$date_dir" | gcloud storage cp - "${inputs_uri%/}/topo/current"
echo "topo/current now points to ${date_dir}"
