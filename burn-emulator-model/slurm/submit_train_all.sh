#!/bin/bash -l
set -euo pipefail

# run from the repo root: burn-emulator-model/slurm/submit_train_all.sh [node ...]
# varlocs.txt indices are dealt round-robin across the nodes, one array job per node

NODES=("$@")
[ ${#NODES[@]} -eq 0 ] && NODES=(dragon02 dragon04 dragon05 dragon06)

VARLOCS_FILE=burn-emulator-model/configs/varlocs/varlocs.txt
N_VARLOCS=$(grep -cvE '^[[:space:]]*$' "$VARLOCS_FILE")
echo "$N_VARLOCS varlocs across ${#NODES[@]} nodes: ${NODES[*]}"

for n in "${!NODES[@]}"; do
    NODE=${NODES[$n]}
    case "$NODE" in
        dragon02) SLOTS=6 ;;
        dragon04|dragon05|dragon06) SLOTS=8 ;;
        *) echo "unknown node: $NODE" >&2; exit 1 ;;
    esac

    INDICES=$(seq -s, "$n" "${#NODES[@]}" $((N_VARLOCS - 1)))
    if [ -z "$INDICES" ]; then
        echo "[$NODE] no varlocs left, skipping"
        continue
    fi

    NODE_INFO=$(scontrol show node "$NODE")
    EFCT_CPUS=$(grep -oP 'CPUEfctv=\K[0-9]+' <<<"$NODE_INFO")
    TOTAL_MEM_MB=$(grep -oP 'RealMemory=\K[0-9]+' <<<"$NODE_INFO")
    CPUS_PER_TASK=$((EFCT_CPUS / SLOTS))
    MEM_PER_TASK=$((TOTAL_MEM_MB / SLOTS))

    echo "[$NODE] SLOTS=$SLOTS indices=$INDICES cpus-per-task=$CPUS_PER_TASK mem=$MEM_PER_TASK"

    sbatch \
        --nodelist="$NODE" \
        --array="${INDICES}%${SLOTS}" \
        --cpus-per-task="$CPUS_PER_TASK" \
        --mem="$MEM_PER_TASK" \
        --export=ALL,SLOTS="$SLOTS" \
        "$(dirname "$0")/train_varlocs.slurm"
done
