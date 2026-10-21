#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/load_env.sh"

# uploads <inputs_root>/<current>/ as is, then repoints <inputs_uri>/current
inputs_root="${1:-}"
inputs_uri="${2:-${BURN_EMULATOR_INPUTS_URI:-}}"

if [[ -z "$inputs_root" ]]; then
    echo "usage: $0 <inputs_root> [inputs_uri]" >&2
    exit 2
fi
if [[ -z "$inputs_uri" ]]; then
    echo "error: pass inputs_uri as arg 2, or run make env-vars" >&2
    exit 2
fi
if [[ ! -f "${inputs_root%/}/current" ]]; then
    echo "error: ${inputs_root%/}/current not found, run 'make inputs-current'" >&2
    exit 1
fi
data_version="$(<"${inputs_root%/}/current")"
inputs_dir="${inputs_root%/}/$data_version"
fbfm_copy="$inputs_dir/fbfm/fbfm_behavior_adjectives.csv"
for f in "$fbfm_copy" "$inputs_dir/varlocs/all_varlocs.gpkg"; do
    [[ -f "$f" ]] || { echo "error: $f not found" >&2; exit 1; }
done
[[ -f "$inputs_dir/varlocs/valid_varlocs.gpkg" ]] || { echo "error: $inputs_dir/varlocs/valid_varlocs.gpkg not found, run 'make valid-varlocs'" >&2; exit 1; }

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

publish_layer baseline "$inputs_dir"/baseline/*.tif
publish_layer legalmax "$inputs_dir"/legalmax/*.tif
publish_layer topo "$inputs_dir"/topo/*.tif
publish_layer varlocs "$inputs_dir"/varlocs/*.gpkg
publish_layer fbfm "$fbfm_copy"

# repoints current after every layer is up
printf '%s' "$data_version" | gcloud storage cp - "${inputs_uri%/}/current"
echo "current now points to ${data_version}"
