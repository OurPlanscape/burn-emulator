#!/usr/bin/env bash
#
# publish_model.sh — publish a trained model bundle to the GCS model registry.
#
#   scripts/publish_model.sh <varloc> <bundle_dir> [models_uri]
#
# <bundle_dir> is a local directory holding one model version, as produced by
# `burn_emulator -m bundle`:
#
#   model.pt                 the checkpoint
#   config.yaml              merged model + activation + dataset + dataloader
#   topo/  stats.yaml  ...    static inputs the config references
#
# It gets uploaded to:
#
#   <models_uri>/<varloc>/<version>/     the bundle
#   <models_uri>/<varloc>/current        one-line pointer, repointed to <version>
#
# <version> is "<checkpoint-mtime>-<short-git-sha>", e.g. 20260829T143005Z-a1b2c3d.
# Keying off the checkpoint's mtime makes re-publishing the same file idempotent.
#
# Needs: gcloud, and GNU coreutils (stat, date).

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

# version = when the checkpoint was written + the model code it came from
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
git_sha="$(git -C "$repo_root" rev-parse --short HEAD)"
git -C "$repo_root" diff --quiet || git_sha="${git_sha}-dirty"

checkpoint_mtime="$(stat -c %Y "$bundle_dir/model.pt")"
timestamp="$(date -u -d "@${checkpoint_mtime}" +%Y%m%dT%H%M%SZ)"
version="${timestamp}-${git_sha}"

base="${models_uri%/}/${varloc}"

echo "varloc   ${varloc}"
echo "version  $${version}"
echo "from     ${bundle_dir}"
echo "to       ${base}/$${version}/"
echo

gcloud storage cp --recursive "${bundle_dir%/}/"* "${base}/$${version}/"
printf '%s' "$version" | gcloud storage cp - "${base}/current"

echo
echo "done — ${varloc}/current now points to $${version}"
