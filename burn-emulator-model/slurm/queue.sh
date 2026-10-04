# shared work queue for the slurm worker arrays (train.slurm, ignitions.slurm); source with QUEUE set.
# one line per varloc, rewritten in place as workers go:
#   <varloc>                                         pending (delete the line to drop it)
#   <varloc> running <job> <node> <start>            claimed
#   <varloc> ok|failed <job> <node> <start> <end>    finished (cut back to <varloc> to requeue)

# queue_edit <varloc> <awk action> [r]: rewrite the varloc's line under the queue lock (t = now);
# written back in place so the file keeps its inode for tail -f / editors
queue_edit () {
    (
        flock 9
        awk -v v="$1" -v r="${3:-}" -v t="$(date +%FT%T)" '$1 == v && !done { done = 1; '"$2"'; next } { print }' \
            "$QUEUE" > "$QUEUE.tmp" && cat "$QUEUE.tmp" > "$QUEUE" && rm -f "$QUEUE.tmp"
    ) 9>>"$QUEUE.lock"
}

# claim the first pending (bare) line; prints its varloc, or nothing once none are left
queue_claim () {
    (
        flock 9
        local v
        v=$(awk 'NF == 1 { print $1; exit }' "$QUEUE")
        [ -n "$v" ] || exit 0
        awk -v v="$v" -v s="running $SLURM_JOB_ID $SLURMD_NODENAME $(date +%FT%T)" \
            '$1 == v && NF == 1 && !done { done = 1; print v, s; next } { print }' \
            "$QUEUE" > "$QUEUE.tmp" && cat "$QUEUE.tmp" > "$QUEUE" && rm -f "$QUEUE.tmp"
        echo "$v"
    ) 9>>"$QUEUE.lock"
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
