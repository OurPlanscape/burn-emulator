#!/bin/bash -l
set -euo pipefail

usage () { echo "usage: $0 <dragon02|dragon04|dragon05> [array-range, e.g. 0-12]" >&2; exit 1; }

NODE=${1:-}
ARRAY_RANGE=${2:-0-12}
[ -z "$NODE" ] && usage

case "$NODE" in
    dragon02) SLOTS=6 ;;
    dragon04|dragon05) SLOTS=8 ;;
    *) echo "unknown node: $NODE" >&2; usage ;;
esac

NODE_INFO=$(scontrol show node "$NODE")
TOTAL_CPUS=$(grep -oP 'CPUTot=\K[0-9]+' <<<"$NODE_INFO")
TOTAL_MEM_MB=$(grep -oP 'RealMemory=\K[0-9]+' <<<"$NODE_INFO")

CPUS_PER_TASK=$((TOTAL_CPUS / SLOTS))
MEM_PER_TASK=$((TOTAL_MEM_MB / SLOTS))

echo "[$NODE] SLOTS=$SLOTS total_cpus=$TOTAL_CPUS total_mem_mb=$TOTAL_MEM_MB -> cpus-per-task=$CPUS_PER_TASK mem=$MEM_PER_TASK"

sbatch \
    --nodelist="$NODE" \
    --array="${ARRAY_RANGE}%${SLOTS}" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM_PER_TASK" \
    --export=ALL,SLOTS="$SLOTS" \
    "$(dirname "$0")/slurm/train_varlocs.slurm"
