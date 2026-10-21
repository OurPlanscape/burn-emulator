#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/load_env.sh"

models_uri="${BURN_EMULATOR_MODELS_URI:-}"
# <bundles_root>/<VARLOC>/{current,<model_version>/}, written by burn_emulator -m bundle (default: this checkout's data/bundles)
bundles_root="${BURN_EMULATOR_BUNDLES_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/bundles}"

if [[ $# -eq 0 ]]; then
    echo "usage: $0 <varloc> [<varloc> ...]" >&2
    exit 2
fi
if [[ -z "$models_uri" ]]; then
    echo "error: BURN_EMULATOR_MODELS_URI is not set - run make env-vars" >&2
    exit 2
fi

# validate every local current bundle before touching the registry
declare -A seen=()
varlocs=()
model_versions=()
for varloc in "$@"; do
    varloc="${varloc^^}"
    if [[ -n "${seen[$varloc]:-}" ]]; then
        echo "error: varloc $varloc passed more than once" >&2
        exit 2
    fi
    seen[$varloc]=1

    pointer="$bundles_root/$varloc/current"
    if [[ ! -f "$pointer" ]]; then
        echo "error: $pointer not found, run 'make model-bundle VARLOC=$varloc'" >&2
        exit 1
    fi
    model_version="$(<"$pointer")"
    bundle_dir="$bundles_root/$varloc/$model_version"
    for f in model.pt config.yaml stats.yaml bundle_meta.json; do
        if [[ ! -f "$bundle_dir/$f" ]]; then
            echo "error: $bundle_dir is missing $f, re-run 'make model-bundle VARLOC=$varloc'" >&2
            exit 1
        fi
    done

    model_name="$(grep -oP '^model_name:[[:space:]]*\K\S+' "$bundle_dir/config.yaml" || true)"
    if [[ "$model_name" != "${varloc}_"* ]]; then
        echo "error: $bundle_dir is a bundle for '${model_name:-unknown}', not varloc '$varloc'" >&2
        exit 1
    fi

    varlocs+=("$varloc")
    model_versions+=("$model_version")
done

# a different bundle under a published version is refused unless FORCE=1
for i in "${!varlocs[@]}"; do
    varloc="${varlocs[$i]}"
    model_version="${model_versions[$i]}"
    bundle_dir="$bundles_root/$varloc/$model_version"
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
        echo
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
