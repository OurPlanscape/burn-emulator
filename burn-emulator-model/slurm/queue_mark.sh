#!/bin/bash
set -euo pipefail

# mark queue lines safely while workers run (takes the same lock they do):
#   burn-emulator-model/slurm/queue_mark.sh <queue file> skip <varloc>...      skip pending varlocs
#   burn-emulator-model/slurm/queue_mark.sh <queue file> pending <varloc>...   requeue (cut back to <varloc>)
# running varlocs are left alone: cancel their job (third field) instead

[ $# -ge 3 ] || { echo "usage: $0 <queue file> skip|pending <varloc>..." >&2; exit 1; }
QUEUE=$(realpath "$1")
state=$2
shift 2
source "$(dirname "$0")/queue.sh"

# the state check runs inside the lock, so a worker cannot claim the line in between
case "$state" in
    skip) action='if (NF == 1) print $1, "skip"; else print' ;;
    pending) action='if ($2 == "running") print; else print $1' ;;
    *) echo "unknown state $state (skip|pending)" >&2; exit 1 ;;
esac
for varloc in "$@"; do
    grep -qE "^[[:space:]]*$varloc([[:space:]]|\$)" "$QUEUE" || { echo "$varloc: not in the queue" >&2; continue; }
    queue_edit "$varloc" "$action"
    awk -v v="$varloc" '$1 == v { print; exit }' "$QUEUE"
done
