#!/usr/bin/env bash

set -euo pipefail

varloc="${1:-}"
varloc="${varloc^^}"
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

# the bundle must be complete before the script touches the registry
for f in model.pt config.yaml stats.yaml bundle_meta.json; do
    if [[ ! -f "$bundle_dir/$f" ]]; then
        echo "error: $bundle_dir is missing $f, re-run 'burn_emulator -m bundle'" >&2
        exit 1
    fi
done
fbfm_map="$(grep -oP '^[[:space:]]*fbfm_map_path:[[:space:]]*\K\S+' "$bundle_dir/config.yaml" || true)"
if [[ -z "$fbfm_map" || ! -f "$bundle_dir/$fbfm_map" ]]; then
    echo "error: $bundle_dir is missing the fbfm map '${fbfm_map:-unset}' named in config.yaml" >&2
    exit 1
fi

model_name="$(grep -oP '^model_name:[[:space:]]*\K\S+' "$bundle_dir/config.yaml" || true)"
if [[ "$model_name" != "${varloc}_"* ]]; then
    echo "error: $bundle_dir is a bundle for '${model_name:-unknown}', not varloc '$varloc'" >&2
    exit 1
fi

# model_version = when the checkpoint was written + the model code it was bundled from
meta="$bundle_dir/bundle_meta.json"
repo_sha="$(grep -oP '"model_repo_sha":[[:space:]]*"\K[0-9a-f]+' "$meta" || true)"
if [[ -z "$repo_sha" ]]; then
    echo "error: $meta has no model_repo_sha (bundled outside git?)" >&2
    exit 1
fi
git_sha="${repo_sha:0:7}"
grep -qP '"model_repo_dirty":[[:space:]]*true' "$meta" && git_sha="${git_sha}-dirty"

checkpoint_mtime="$(stat -c %Y "$bundle_dir/model.pt")"
timestamp="$(date -u -d "@${checkpoint_mtime}" +%Y%m%dT%H%M%SZ)"
model_version="${timestamp}-${git_sha}"

base="${models_uri%/}/${varloc}"

echo "varloc         ${varloc}"
echo "model_version  ${model_version}"
echo "from           ${bundle_dir}"
echo "to             ${base}/${model_version}/"
echo

# outputs are cached per model_version, so never silently replace a published
# version: an identical re-publish just repoints current, a different bundle
# under the same version is refused unless FORCE=1
dest="${base}/${model_version}"
if [[ "${FORCE:-0}" != "1" ]] && gcloud storage ls "${dest}/" >/dev/null 2>&1; then
    for f in "${bundle_dir%/}"/*; do
        name="$(basename "$f")"
        [[ "$name" == model.pt ]] && continue
        if ! gcloud storage cat "${dest}/${name}" 2>/dev/null | cmp -s - "$f"; then
            echo "error: ${dest}/ already exists with a different ${name}; rebundle from a new commit, or FORCE=1 to overwrite (cached outputs under it are NOT invalidated)" >&2
            exit 1
        fi
    done
    echo "already published: ${dest}/ matches this bundle, skipping upload"
    echo
else
    gcloud storage cp --recursive "${bundle_dir%/}/"* "${dest}/"
fi
printf '%s' "$model_version" | gcloud storage cp - "${base}/current"

echo
echo "done: ${varloc}/current now points to ${model_version}"
