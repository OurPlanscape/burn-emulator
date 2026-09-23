#!/usr/bin/env bash

set -euo pipefail

layer_dir="${1:-}"
fuels_uri="${2:-${BURN_EMULATOR_FUELS_URI:-}}"

if [[ -z "$layer_dir" ]]; then
    echo "usage: $0 <layer_dir> [fuels_uri]" >&2
    exit 2
fi
if [[ -z "$fuels_uri" ]]; then
    echo "error: pass fuels_uri as arg 2, or set BURN_EMULATOR_FUELS_URI" >&2
    exit 2
fi

# layer_dir holds both baseline and legalmax tifs together, split by filename
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

base="${fuels_uri%/}/${date_dir}"

baseline_files=()
legalmax_files=()
unmatched_files=()
for f in "$layer_dir"/*.tif; do
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
    echo "error: filenames must contain 'baseline' or 'legalmax', got: ${unmatched_files[*]}" >&2
    exit 1
fi
if [[ ${#baseline_files[@]} -eq 0 ]]; then
    echo "error: no *baseline*.tif files found in $layer_dir" >&2
    exit 1
fi
if [[ ${#legalmax_files[@]} -eq 0 ]]; then
    echo "error: no *legalmax*.tif files found in $layer_dir" >&2
    exit 1
fi

# dedup: skip a treatment that's already published (FORCE=1 to re-upload)
publish_treatment () {
    local treatment="$1"
    shift
    local dest="${base}/${treatment}"

    echo "layer    ${treatment}"
    echo "date     ${date_dir}"
    echo "from     ${layer_dir} ($# files)"
    echo "to       ${dest}/"
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
    echo "done: ${treatment} layer published to ${dest}/"
    echo
}

publish_treatment baseline "${baseline_files[@]}"
publish_treatment legalmax "${legalmax_files[@]}"

printf '%s' "$date_dir" | gcloud storage cp - "${fuels_uri%/}/fuels/current"
echo "fuels/current now points to ${date_dir}"
