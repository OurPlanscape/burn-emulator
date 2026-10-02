#!/bin/bash -l
set -euo pipefail

# run from the repo root: burn-emulator-model/slurm/submit_train_all.sh [-p] [-v <varloc>]... [-r <indices>] [node[,node...] ...]
# trains every trainable varloc (-v: only these, -r: indices e.g. 0-5,9) across the given GPU nodes
# -p also bundles + publishes each (needs BURN_EMULATOR_MODELS_URI and BURN_EMULATOR_INPUTS_URI)

usage () { echo "usage: $0 [-p] [-v <varloc>]... [-r <indices, e.g. 0-5,9>] [node[,node...] ...]" >&2; exit 1; }

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

# node's GPU count from slurm GRES, 0 if none
node_gpus () { scontrol show node "$1" 2>/dev/null | grep -oP 'Gres=gpu:([^:,(]+:)?\K[0-9]+' || echo 0; }
SHARED_SLOTS=8

# nodes as args and/or comma-separated
NODES=()
for arg in "$@"; do
    IFS=, read -ra PARTS <<<"$arg"
    for part in "${PARTS[@]}"; do
        [ -z "$part" ] || [[ " ${NODES[*]} " == *" $part "* ]] || NODES+=("$part")
    done
done
[ ${#NODES[@]} -eq 0 ] && NODES=(dragon02 dragon04 dragon05 dragon06)
declare -A GPUS
POOL=()
POOL_GPUS=0
for NODE in "${NODES[@]}"; do
    GPUS[$NODE]=$(node_gpus "$NODE")
    case "${GPUS[$NODE]}" in
        0) echo "$NODE has no GPUs in slurm (or does not exist)" >&2; exit 1 ;;
        1) ;;
        *) POOL+=("$NODE"); POOL_GPUS=$((POOL_GPUS + GPUS[$NODE])) ;;
    esac
done

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

# every worker claims varlocs from one shared queue
STAMP=$(date +%Y%m%dT%H%M%S)
QUEUE=$(realpath burn-emulator-model/data/logs)/train_queue_${STAMP}.txt
for i in "${INDICES[@]}"; do
    printf '%s\n' "${VARLOCS[$i]}"
done > "$QUEUE"
mkdir "${QUEUE%.txt}.d"
echo "queue: $QUEUE"

# submit_array <name> <slots> <workers> <cpus-per-task> <mem> [sbatch args...]
submit_array () {
    local name=$1 slots=$2 workers=$3 cpus=$4 mem=$5
    shift 5
    ARRAY_ID=$(sbatch --parsable "$@" \
        --array="0-$((workers - 1))" \
        --job-name="train_$name" \
        --cpus-per-task="$cpus" \
        --mem="$mem" \
        --export=ALL,VARLOC=,QUEUE="$QUEUE",SLOTS="$slots",PUBLISH="$PUBLISH",MONITOR="$MONITOR" \
        "$(dirname "$0")/train.slurm")
    ARRAY_ID=${ARRAY_ID%%;*}
    MONITOR=""
    echo "[$name] array $ARRAY_ID: $workers workers slots=$slots cpus-per-task=$cpus mem=$mem $*"
}

# per-worker cpus and mem: the node's total / slots
node_share () {
    local info
    info=$(scontrol show node "$1")
    echo "$(($(grep -oP 'CPUEfctv=\K[0-9]+' <<<"$info") / $2)) $(($(grep -oP 'RealMemory=\K[0-9]+' <<<"$info") / $2))"
}

REMAINING=${#INDICES[@]}
MONITOR=1
POOL_DONE=""
for NODE in "${NODES[@]}"; do
    [ "$REMAINING" -gt 0 ] || break
    if [ "${GPUS[$NODE]}" -gt 1 ]; then
        # one array for all multi-GPU nodes, sized to fit the smallest per-GPU share
        [ -z "$POOL_DONE" ] || continue
        POOL_DONE=1
        WORKERS=$((REMAINING < POOL_GPUS ? REMAINING : POOL_GPUS))
        REMAINING=$((REMAINING - WORKERS))
        CPUS_PER_TASK="" MEM_PER_TASK=""
        for P in "${POOL[@]}"; do
            read -r c m <<<"$(node_share "$P" "${GPUS[$P]}")"
            [ -z "$CPUS_PER_TASK" ] || [ "$c" -lt "$CPUS_PER_TASK" ] && CPUS_PER_TASK=$c
            [ -z "$MEM_PER_TASK" ] || [ "$m" -lt "$MEM_PER_TASK" ] && MEM_PER_TASK=$m
        done
        # --exclude, as a multi-node --nodelist would require every node per task
        EXCLUDE=$(sinfo -h -N -o %N | sort -u | { grep -vxF -f <(printf '%s\n' "${POOL[@]}") || true; } | paste -sd, -)
        POOL_NAME=$(IFS=-; echo "${POOL[*]}")
        submit_array "$POOL_NAME" "$POOL_GPUS" "$WORKERS" "$CPUS_PER_TASK" "$MEM_PER_TASK" \
            --gres=gpu:1 ${EXCLUDE:+--exclude="$EXCLUDE"}
    else
        WORKERS=$((REMAINING < SHARED_SLOTS ? REMAINING : SHARED_SLOTS))
        REMAINING=$((REMAINING - WORKERS))
        read -r CPUS_PER_TASK MEM_PER_TASK <<<"$(node_share "$NODE" "$SHARED_SLOTS")"
        submit_array "$NODE" "$SHARED_SLOTS" "$WORKERS" "$CPUS_PER_TASK" "$MEM_PER_TASK" --nodelist="$NODE"
    fi
done
