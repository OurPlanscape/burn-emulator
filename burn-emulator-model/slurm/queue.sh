# shared work queue for the slurm worker arrays (train.slurm, ignitions.slurm); source with QUEUE set.
# one line per varloc, rewritten in place as workers go:
#   <varloc>                                         pending (delete the line to drop it)
#   <varloc> running <job> <node> <start>            claimed
#   <varloc> ok|failed <job> <node> <start> <end>    finished (cut back to <varloc> to requeue)

queue_lock () {
    local waited=0
    until mkdir "$QUEUE.lockdir" 2>/dev/null; do
        if [ "$waited" -ge 300 ]; then
            echo "warning: breaking stale queue lock $QUEUE.lockdir" >&2
            rmdir "$QUEUE.lockdir" 2>/dev/null || true
            waited=0
        fi
        sleep 0.2
        waited=$((waited + 1))
    done
}
queue_unlock () { rmdir "$QUEUE.lockdir"; }

# rewrites the queue from stdin in place (same inode)
queue_write () { cat > "$QUEUE.tmp" && cat "$QUEUE.tmp" > "$QUEUE" && rm -f "$QUEUE.tmp"; }

# queue_edit <varloc> <awk action> [r]: rewrite the varloc's line under the queue lock (t = now)
queue_edit () {
    queue_lock
    awk -v v="$1" -v r="${3:-}" -v t="$(date +%FT%T)" '$1 == v && !done { done = 1; '"$2"'; next } { print }' \
        "$QUEUE" | queue_write
    queue_unlock
}

# claim the first pending (bare) line; prints its varloc, or nothing once none are left
queue_claim () {
    local v
    queue_lock
    v=$(awk 'NF == 1 { print $1; exit }' "$QUEUE")
    if [ -n "$v" ]; then
        awk -v v="$v" -v s="running $SLURM_JOB_ID $SLURMD_NODENAME $(date +%FT%T)" \
            '$1 == v && NF == 1 && !done { done = 1; print v, s; next } { print }' "$QUEUE" | queue_write
    fi
    queue_unlock
    echo "$v"
}

# queue_finish <varloc> ok|failed
queue_finish () { queue_edit "$1" '$2 = r; print $0, t' "$2"; }

# queue_run <log prefix> <fn>: claim varlocs until none are pending, running <fn> <varloc> for each with
# its output in data/logs/<prefix>_<varloc>_<job>.{out,err}; returns 1 if any failed
queue_run () {
    local prefix=$1 fn=$2 varloc result status=0
    while varloc=$(queue_claim) && [ -n "$varloc" ]; do
        echo "[queue] $varloc claimed at $(date)"
        result=ok
        "$fn" "$varloc" < /dev/null \
            > "data/logs/${prefix}_${varloc}_${SLURM_JOB_ID}.out" \
            2> "data/logs/${prefix}_${varloc}_${SLURM_JOB_ID}.err" || { result=failed; status=1; }
        queue_finish "$varloc" "$result"
        echo "[queue] $varloc $result at $(date)"
    done
    return "$status"
}
