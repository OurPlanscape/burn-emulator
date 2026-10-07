#!/bin/bash
set -euo pipefail

# rewrites data/outputs/varlocs.txt to the varlocs with an epoch-<num_epochs - 1> checkpoint for the
# current architecture + data_version, under mark_trained.sh's lock. run from burn-emulator-model/

varlocs_txt=data/outputs/varlocs.txt

arch=$(grep -oP '^architecture:[[:space:]]*\K\S+' configs/varlocs/current.yaml)
dv=$(scripts/data_version.sh)
epochs=$(grep -oP '^num_epochs:[[:space:]]*\K[0-9]+' "configs/$arch/train.yaml" || true)
if [ -z "$epochs" ]; then
    echo "error: no num_epochs in configs/$arch/train.yaml" >&2
    exit 2
fi
last=$(printf '%04d' $((epochs - 1)))

exec {lock_fd}>"$varlocs_txt.lock"
flock "$lock_fd"
touch "$varlocs_txt"

trained=()
for dir in data/outputs/*_"${arch}_${dv}"/; do
    [ -d "$dir" ] || continue
    model_name=$(basename "$dir")
    varloc=${model_name%%_*}
    if compgen -G "${dir}checkpoints/${model_name}_*_epoch-${last}_step-*.pt" >/dev/null; then
        trained+=("${varloc^^}")
    else
        echo "not trained: $varloc (no epoch-$last checkpoint in ${dir}checkpoints)"
    fi
done

if [ ${#trained[@]} -eq 0 ]; then
    echo "error: no ${arch}_${dv} run has an epoch-$last checkpoint; leaving $varlocs_txt unchanged" >&2
    exit 1
fi

printf '%s\n' "${trained[@]}" | LC_ALL=C sort -u > "$varlocs_txt.tmp"
diff <(grep -vE '^[[:space:]]*$' "$varlocs_txt" | LC_ALL=C sort -u) "$varlocs_txt.tmp" | grep -E '^[<>]' \
    | sed 's/^</removed/; s/^>/added  /' || true
mv "$varlocs_txt.tmp" "$varlocs_txt"
echo "$varlocs_txt: ${#trained[@]} varlocs for ${arch}_${dv}"
