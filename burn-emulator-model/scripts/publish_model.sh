#!/usr/bin/env bash

set -euo pipefail

models_uri="${BURN_EMULATOR_MODELS_URI:-}"

if [[ $# -eq 0 ]]; then
    echo "usage: BURN_EMULATOR_MODELS_URI=<gs://...> $0 <varloc>=<bundle_dir> [<varloc>=<bundle_dir> ...]" >&2
    exit 2
fi
if [[ -z "$models_uri" ]]; then
    echo "error: set BURN_EMULATOR_MODELS_URI" >&2
    exit 2
fi

# validate every bundle before touching the registry
declare -A seen=()
varlocs=()
bundle_dirs=()
model_versions=()
for pair in "$@"; do
    varloc="${pair%%=*}"
    varloc="${varloc^^}"
    bundle_dir="${pair#*=}"
    if [[ "$pair" != *=* || -z "$varloc" || -z "$bundle_dir" ]]; then
        echo "error: '$pair' is not <varloc>=<bundle_dir>" >&2
        exit 2
    fi
    if [[ -n "${seen[$varloc]:-}" ]]; then
        echo "error: varloc $varloc passed more than once" >&2
        exit 2
    fi
    seen[$varloc]=1

    for f in model.pt config.yaml stats.yaml bundle_meta.json; do
        if [[ ! -f "$bundle_dir/$f" ]]; then
            echo "error: $bundle_dir is missing $f, re-run 'burn_emulator -m bundle'" >&2
            exit 1
        fi
    done

    model_name="$(grep -oP '^model_name:[[:space:]]*\K\S+' "$bundle_dir/config.yaml" || true)"
    if [[ "$model_name" != "${varloc}_"* ]]; then
        echo "error: $bundle_dir is a bundle for '${model_name:-unknown}', not varloc '$varloc'" >&2
        exit 1
    fi

    # model_version = checkpoint mtime + bundled repo sha
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

    varlocs+=("$varloc")
    bundle_dirs+=("$bundle_dir")
    model_versions+=("${timestamp}-${git_sha}")
done

# a different bundle under a published version is refused unless FORCE=1
for i in "${!varlocs[@]}"; do
    varloc="${varlocs[$i]}"
    bundle_dir="${bundle_dirs[$i]}"
    model_version="${model_versions[$i]}"
    dest="${models_uri%/}/${varloc}/${model_version}"

    echo "varloc         ${varloc}"
    echo "model_version  ${model_version}"
    echo "from           ${bundle_dir}"
    echo "to             ${dest}/"
    echo

    if [[ "${FORCE:-0}" != "1" ]] && gcloud storage ls "${dest}/" >/dev/null 2>&1; then
        for f in "${bundle_dir%/}"/*; do
            name="$(basename "$f")"
            [[ "$name" == model.pt ]] && continue
            if ! gcloud storage cat "${dest}/${name}" 2>/dev/null | cmp -s - "$f"; then
                echo "error: ${dest}/ already exists with a different ${name}; rebundle from a new commit, or FORCE=1 to overwrite" >&2
                exit 1
            fi
        done
        echo "already published: ${dest}/ matches this bundle, skipping upload"
        echo
    else
        gcloud storage cp --recursive "${bundle_dir%/}/"* "${dest}/"
    fi
done

# repoints each <varloc>/current after every bundle is up
for i in "${!varlocs[@]}"; do
    pointer="${models_uri%/}/${varlocs[$i]}/current"
    if [[ "$(gcloud storage cat "$pointer" 2>/dev/null || true)" == "${model_versions[$i]}" ]]; then
        echo "already current: ${varlocs[$i]} = ${model_versions[$i]}"
        continue
    fi
    printf '%s' "${model_versions[$i]}" | gcloud storage cp - "$pointer"
    echo "current: ${varlocs[$i]} = ${model_versions[$i]}"
done
