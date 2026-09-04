#!/usr/bin/env bash

set -euo pipefail

layer_dir="${1:-}"
layer="${2:-}"
fuels_uri="${3:-${BURN_EMULATOR_FUELS_URI:-}}"

if [[ -z "$layer_dir" || -z "$layer" ]]; then
    echo "usage: $0 <layer_dir> <layer> [fuels_uri]" >&2
    exit 2
fi
case "$layer" in
    baseline | legalmax | topo) ;;
    *)
        echo "error: <layer> must be one of: baseline legalmax topo (got '$layer')" >&2
        exit 2
        ;;
esac
if [[ -z "$fuels_uri" ]]; then
    echo "error: pass fuels_uri as arg 3, or set BURN_EMULATOR_FUELS_URI" >&2
    exit 2
fi

# the layer must be a real directory of tifs before we touch the registry
if [[ ! -d "$layer_dir" ]]; then
    echo "error: $layer_dir is not a directory" >&2
    exit 1
fi
if ! compgen -G "$layer_dir/*.tif" >/dev/null; then
    echo "error: $layer_dir has no *.tif layers" >&2
    exit 1
fi

# date = the DDMonYYYY stamp on the directory name, normalised to ISO YYYYMMDD
dir_name="$(basename "${layer_dir%/}")"
if [[ ! "$dir_name" =~ ([0-9]{1,2})([A-Za-z]{3})([0-9]{4}) ]]; then
    echo "error: no DDMonYYYY date in directory name '$dir_name'" >&2
    exit 1
fi
date_dir="$(date -u -d "${BASH_REMATCH[1]} ${BASH_REMATCH[2]} ${BASH_REMATCH[3]}" +%Y%m%d)"

base="${fuels_uri%/}/${date_dir}/${layer}"

echo "layer    ${layer}"
echo "date     ${date_dir}"
echo "from     ${layer_dir}"
echo "to       ${base}/"
echo

# dedup: skip a layer directory that already exists (FORCE=1 to re-upload)
cp_flags=(--recursive --no-clobber)
if [[ "${FORCE:-0}" == "1" ]]; then
    cp_flags=(--recursive)
elif gcloud storage ls "${base}/" >/dev/null 2>&1; then
    echo "already published: ${base}/ exists (FORCE=1 to re-upload)"
    exit 0
fi

gcloud storage cp "${cp_flags[@]}" "${layer_dir%/}/"* "${base}/"

echo
echo "done: ${layer} layer published to ${base}/"
