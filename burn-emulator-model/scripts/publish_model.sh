#!/usr/bin/env bash

set -euo pipefail

varloc="${1:-}"
bundle_dir="${2:-}"
models_uri="${3:-${BURN_EMULATOR_MODELS_URI:-}}"

if [[ -z "$varloc" || -z "$bundle_dir" ]]; then
    echo "usage: $0 <varloc> <bundle_dir> [models_uri]" >&2
    exit 2
fi
if [[ -z "$models_uri" ]]; then
    echo "error: pass models_uri as arg 3, or set BURN_EMULATOR_MODELS_URI" >&2
    exit 2
fi

# the bundle must be complete before we touch the registry
if [[ ! -f "$bundle_dir/model.pt" ]]; then
    echo "error: $bundle_dir is missing model.pt" >&2
    exit 1
fi
if ! compgen -G "$bundle_dir/*.yaml" >/dev/null; then
    echo "error: $bundle_dir has no *.yaml config" >&2
    exit 1
fi
if [[ ! -f "$bundle_dir/bundle_meta.json" ]]; then
    echo "error: $bundle_dir is missing bundle_meta.json — re-run 'burn_emulator -m bundle'" >&2
    exit 1
fi

# version = when the checkpoint was written + the model code it came from
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
git_sha="$(git -C "$repo_root" rev-parse --short HEAD)"
git -C "$repo_root" diff --quiet || git_sha="${git_sha}-dirty"

checkpoint_mtime="$(stat -c %Y "$bundle_dir/model.pt")"
timestamp="$(date -u -d "@${checkpoint_mtime}" +%Y%m%dT%H%M%SZ)"
version="${timestamp}-${git_sha}"

base="${models_uri%/}/${varloc}"

echo "varloc   ${varloc}"
echo "version  ${version}"
echo "from     ${bundle_dir}"
echo "to       ${base}/${version}/"
echo

gcloud storage cp --recursive "${bundle_dir%/}/"* "${base}/${version}/"
printf '%s' "$version" | gcloud storage cp - "${base}/current"

echo
echo "done — ${varloc}/current now points to ${version}"
