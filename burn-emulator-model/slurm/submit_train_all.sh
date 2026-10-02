#!/bin/bash -l
set -euo pipefail

# run from the repo root: burn-emulator-model/slurm/submit_train_all.sh [-p] [-v <varloc>]... [-r <indices>] [node ...]
# indices into the varlocs with complete training data (default: all; -v restricts to the given
# varlocs; -r takes a slurm-style list, e.g. 0-5,9) are dealt round-robin across the nodes, one
# train.slurm array per node with one task per varloc. -p bundles and publishes each varloc once its training succeeds, then
# republishes the varlocs txt + gpkg (needs BURN_EMULATOR_MODELS_URI and BURN_EMULATOR_INPUTS_URI exported)

usage () { echo "usage: $0 [-p] [-v <varloc>]... [-r <indices, e.g. 0-5,9>] [node ...]" >&2; exit 1; }

RANGE=""
PUBLISH=""
SELECTED=()
while getopts "pv:r:" opt; do
    case "$opt" in
        p) PUBLISH=1 ;;
        v) SELECTED+=("$OPTARG") ;;
        r) RANGE=$OPTARG ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))

if [ -n "$PUBLISH" ]; then
    for var in BURN_EMULATOR_MODELS_URI BURN_EMULATOR_INPUTS_URI; do
        [ -n "${!var:-}" ] || { echo "error: -p needs $var exported (see README.md)" >&2; exit 2; }
    done
fi

NODES=("$@")
[ ${#NODES[@]} -eq 0 ] && NODES=(dragon02 dragon04 dragon05 dragon06)

mkdir -p burn-emulator-model/data/logs
mapfile -t TRAINABLE < <(cd burn-emulator-model && scripts/trainable_varlocs.sh)
if [ ${#SELECTED[@]} -gt 0 ]; then
    for varloc in "${SELECTED[@]}"; do
        printf '%s\n' "${TRAINABLE[@]}" | grep -qxF "$varloc" || { echo "$varloc has no complete training data" >&2; exit 1; }
    done
    mapfile -t VARLOCS < <(printf '%s\n' "${SELECTED[@]}" | LC_ALL=C sort -u)
else
    VARLOCS=("${TRAINABLE[@]}")
fi
N_VARLOCS=${#VARLOCS[@]}
[ "$N_VARLOCS" -gt 0 ] || { echo "no varlocs with complete training data" >&2; exit 1; }
[ -z "$RANGE" ] && RANGE="0-$((N_VARLOCS - 1))"

INDICES=()
IFS=, read -ra PARTS <<<"$RANGE"
for part in "${PARTS[@]}"; do
    [[ "$part" =~ ^([0-9]+)(-([0-9]+))?$ ]] || { echo "bad range: $part" >&2; usage; }
    lo=${BASH_REMATCH[1]}
    hi=${BASH_REMATCH[3]:-$lo}
    for ((i = lo; i <= hi; i++)); do
        [ "$i" -lt "$N_VARLOCS" ] || { echo "index $i out of range ($N_VARLOCS varlocs)" >&2; exit 1; }
        INDICES+=("$i")
    done
done
mapfile -t INDICES < <(printf '%s\n' "${INDICES[@]}" | sort -nu)

echo "${#INDICES[@]} of $N_VARLOCS varlocs across ${#NODES[@]} nodes: ${NODES[*]}${PUBLISH:+ (bundle + publish)}"

# one held train.slurm array per node (each task takes 1/SLOTS of it, so slurm runs at most SLOTS at
# once per node), tasks renamed to train_<varloc> then released; each array reads a snapshot of its varlocs
STAMP=$(date +%Y%m%dT%H%M%S)
MONITOR=1
for n in "${!NODES[@]}"; do
    NODE=${NODES[$n]}
    case "$NODE" in
        dragon02) SLOTS=6 ;;
        dragon04|dragon05|dragon06) SLOTS=8 ;;
        *) echo "unknown node: $NODE" >&2; exit 1 ;;
    esac

    SUBMIT=()
    for ((k = n; k < ${#INDICES[@]}; k += ${#NODES[@]})); do
        SUBMIT+=("${VARLOCS[${INDICES[$k]}]}")
    done
    [ ${#SUBMIT[@]} -gt 0 ] || continue

    NODE_INFO=$(scontrol show node "$NODE")
    EFCT_CPUS=$(grep -oP 'CPUEfctv=\K[0-9]+' <<<"$NODE_INFO")
    TOTAL_MEM_MB=$(grep -oP 'RealMemory=\K[0-9]+' <<<"$NODE_INFO")
    CPUS_PER_TASK=$((EFCT_CPUS / SLOTS))
    MEM_PER_TASK=$((TOTAL_MEM_MB / SLOTS))

    VARLOCS_LIST=$(realpath burn-emulator-model/data/logs)/train_varlocs_${NODE}_${STAMP}.txt
    printf '%s\n' "${SUBMIT[@]}" > "$VARLOCS_LIST"
    echo "[$NODE] ${#SUBMIT[@]} varlocs SLOTS=$SLOTS cpus-per-task=$CPUS_PER_TASK mem=$MEM_PER_TASK: ${SUBMIT[*]}"

    ARRAY_ID=$(sbatch --parsable --hold \
        --array="0-$((${#SUBMIT[@]} - 1))" \
        --nodelist="$NODE" \
        --cpus-per-task="$CPUS_PER_TASK" \
        --mem="$MEM_PER_TASK" \
        --export=ALL,VARLOC=,VARLOCS_LIST="$VARLOCS_LIST",SLOTS="$SLOTS",PUBLISH="$PUBLISH",MONITOR="$MONITOR" \
        "$(dirname "$0")/train.slurm")
    ARRAY_ID=${ARRAY_ID%%;*}
    MONITOR=""

    for i in "${!SUBMIT[@]}"; do
        scontrol update JobId="${ARRAY_ID}_$i" JobName="train_${SUBMIT[$i]}" \
            || echo "warning: could not rename ${ARRAY_ID}_$i (${SUBMIT[$i]})" >&2
    done
    scontrol release "$ARRAY_ID"
    echo "[$NODE] array $ARRAY_ID: $VARLOCS_LIST"
done
