#!/bin/bash -l
set -euo pipefail

# run from the repo root: burn-emulator-model/slurm/submit_train_all.sh [-r <indices>] [node ...]
# indices into the varlocs with complete training data (default: all; -r takes a
# slurm-style list, e.g. 0-5,9) are dealt round-robin across the nodes, one array job per node

usage () { echo "usage: $0 [-r <indices, e.g. 0-5,9>] [node ...]" >&2; exit 1; }

RANGE=""
while getopts "r:" opt; do
    case "$opt" in
        r) RANGE=$OPTARG ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))

NODES=("$@")
[ ${#NODES[@]} -eq 0 ] && NODES=(dragon02 dragon04 dragon05 dragon06)

# snapshot the list the array indices refer to; tasks read it instead of recomputing
mkdir -p burn-emulator-model/data/logs
VARLOCS_LIST=$(realpath burn-emulator-model/data/logs)/train_varlocs_$(date +%Y%m%dT%H%M%S).txt
(cd burn-emulator-model && scripts/trainable_varlocs.sh) > "$VARLOCS_LIST"
N_VARLOCS=$(grep -cvE '^[[:space:]]*$' "$VARLOCS_LIST" || true)
[ "$N_VARLOCS" -gt 0 ] || { echo "no varlocs with complete training data" >&2; exit 1; }
echo "varlocs snapshot: $VARLOCS_LIST"
[ -z "$RANGE" ] && RANGE="0-$((N_VARLOCS - 1))"

INDICES=()
IFS=, read -ra PARTS <<<"$RANGE"
for part in "${PARTS[@]}"; do
    [[ "$part" =~ ^([0-9]+)(-([0-9]+))?$ ]] || { echo "bad range: $part" >&2; usage; }
    lo=${BASH_REMATCH[1]}
    hi=${BASH_REMATCH[3]:-$lo}
    for ((i = lo; i <= hi; i++)); do
        [ "$i" -lt "$N_VARLOCS" ] || { echo "index $i out of range ($VARLOCS_LIST has $N_VARLOCS)" >&2; exit 1; }
        INDICES+=("$i")
    done
done
mapfile -t INDICES < <(printf '%s\n' "${INDICES[@]}" | sort -nu)

echo "${#INDICES[@]} of $N_VARLOCS varlocs across ${#NODES[@]} nodes: ${NODES[*]}"

for n in "${!NODES[@]}"; do
    NODE=${NODES[$n]}
    case "$NODE" in
        dragon02) SLOTS=6 ;;
        dragon04|dragon05|dragon06) SLOTS=8 ;;
        *) echo "unknown node: $NODE" >&2; exit 1 ;;
    esac

    NODE_INDICES=()
    for ((k = n; k < ${#INDICES[@]}; k += ${#NODES[@]})); do
        NODE_INDICES+=("${INDICES[$k]}")
    done
    if [ ${#NODE_INDICES[@]} -eq 0 ]; then
        echo "[$NODE] no varlocs left, skipping"
        continue
    fi
    ARRAY=$(IFS=,; echo "${NODE_INDICES[*]}")

    NODE_INFO=$(scontrol show node "$NODE")
    EFCT_CPUS=$(grep -oP 'CPUEfctv=\K[0-9]+' <<<"$NODE_INFO")
    TOTAL_MEM_MB=$(grep -oP 'RealMemory=\K[0-9]+' <<<"$NODE_INFO")
    CPUS_PER_TASK=$((EFCT_CPUS / SLOTS))
    MEM_PER_TASK=$((TOTAL_MEM_MB / SLOTS))

    echo "[$NODE] SLOTS=$SLOTS indices=$ARRAY cpus-per-task=$CPUS_PER_TASK mem=$MEM_PER_TASK"

    sbatch \
        --nodelist="$NODE" \
        --array="${ARRAY}%${SLOTS}" \
        --cpus-per-task="$CPUS_PER_TASK" \
        --mem="$MEM_PER_TASK" \
        --export=ALL,SLOTS="$SLOTS",VARLOCS_LIST="$VARLOCS_LIST" \
        "$(dirname "$0")/train_varlocs.slurm"
done
