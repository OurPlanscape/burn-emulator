#!/bin/bash -l
# one dispatcher pass, run every minute from scrontab (slurm/dispatch.scrontab): requeue lost submissions,
# then sbatch slurm/infer.slurm for queued claims. Appends to data/logs/dispatch.log via #SCRON --output.
set -uo pipefail

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/env.sh"
source "$API_DIR/scripts/claims.sh"

: "${INFER_TIME:=00:20:00}"
: "${MAX_INFLIGHT_DL:=2}"
: "${MAX_INFLIGHT_PT:=2}"
: "${SUBMIT_GRACE:=600}"
: "${RUN_STALE_AFTER:=1500}" # dispatch.runStaleAfter, 25 min

now=$(date +%s)
claims=$(claim_list) || { log "error: listing $CLAIMS_URI failed"; exit 1; }

# submitted claims whose slurm job is gone go back to queued; updated_at is kept so they still go stale
requeued=0
while IFS= read -r c; do
    [ -n "$c" ] || continue
    key=$(jq -r .key <<<"$c")
    gen=$(jq -r .generation <<<"$c")
    mgen=$(jq -r .metageneration <<<"$c")
    job=$(jq -r '.metadata.slurm_job // empty' <<<"$c")
    if [ -n "$job" ]; then
        state=$(squeue -h -j "$job" -o %T 2>&1)
        rc=$?
        if [ "$rc" -eq 0 ] && [ -n "$state" ]; then
            continue
        fi
        if [ "$rc" -ne 0 ] && ! grep -qi 'invalid job id' <<<"$state"; then
            log "warn: squeue failed for $key job $job: $state"
            continue
        fi
    elif (( now - $(jq -r '.metadata.submitted_at // 0' <<<"$c") < SUBMIT_GRACE )); then
        continue
    fi
    if claim_update "$key" "$gen" "$mgen" status=queued -submit_id -submitted_at -slurm_job; then
        log "requeue $key (slurm job ${job:-none} gone)"
        requeued=$((requeued + 1))
    fi
done < <(jq -c 'select(.metadata.status == "submitted")' <<<"$claims")

# running claims past the API's stale age (a job that died without a report) stop counting
declare -A inflight=([DL]=0 [PT]=0)
while IFS= read -r key; do
    [ -n "$key" ] || continue
    b=$(claim_backend "$key")
    inflight[$b]=$((inflight[$b] + 1))
done < <(jq -r --argjson now "$now" --argjson stale "$RUN_STALE_AFTER" '
    select(.metadata.status == "submitted"
        or (.metadata.status == "running" and ($now - (.metadata.updated_at | tonumber? // 0)) < $stale))
    | .key' <<<"$claims")

submitted=0
failed=0
queued=0
while IFS= read -r c; do
    [ -n "$c" ] || continue
    queued=$((queued + 1))
    key=$(jq -r .key <<<"$c")
    gen=$(jq -r .generation <<<"$c")
    mgen=$(jq -r .metageneration <<<"$c")
    backend=$(claim_backend "$key")

    args_var=SLURM_${backend}_ARGS
    [ -n "${!args_var:-}" ] || continue
    max_var=MAX_INFLIGHT_$backend
    (( inflight[$backend] < ${!max_var} )) || continue

    body=$(claim_body "$key" "$gen") || { log "warn: reading body of $key failed"; continue; }
    fuels=$(jq -r '.BURN_EMULATOR_BASELINE_FUELS // empty' <<<"$body")
    bundle=$(jq -r '.BURN_EMULATOR_BUNDLE_DIR // empty' <<<"$body")
    reason=""
    if [ -n "$fuels" ] && [ ! -d "$INPUTS_ROOT${fuels#/inputs}" ]; then
        reason="inputs ${fuels#/inputs/} not on cluster (make inputs-release on the cluster)"
    elif [ -n "$bundle" ] && [ ! -f "$BUNDLES_ROOT${bundle#/models}/model.pt" ]; then
        reason="bundle ${bundle#/models/} not on cluster (make model-release on the cluster)"
    fi
    if [ -n "$reason" ]; then
        if claim_update "$key" "$gen" "$mgen" status=failed "error=$reason" "updated_at=$now"; then
            log "fail $key: $reason"
            failed=$((failed + 1))
        fi
        continue
    fi

    submit_id=$(cat /proc/sys/kernel/random/uuid)
    # updated_at is left alone so a claim that never starts still goes stale; infer.slurm resets it on takeover
    claim_update "$key" "$gen" "$mgen" status=submitted "submit_id=$submit_id" "submitted_at=$now" || continue

    read -ra sbatch_args <<<"${!args_var}"
    job=$(sbatch --parsable "${sbatch_args[@]}" \
        --job-name="burn-infer-$backend" \
        --time="$INFER_TIME" \
        --output="$LOG_DIR/infer/%x_%j.out" --error="$LOG_DIR/infer/%x_%j.err" \
        --export="NONE,API_DIR=$API_DIR,CLAIM=$key,CLAIM_GEN=$gen,SUBMIT_ID=$submit_id,BACKEND=$backend" \
        "$API_DIR/slurm/infer.slurm" 2>&1)
    rc=$?
    job=${job%%;*}

    cur=$(claim_get "$key") || cur=""
    cur_gen=$(jq -r '.generation // empty' <<<"$cur")
    cur_mgen=$(jq -r '.metageneration // empty' <<<"$cur")
    cur_sid=$(jq -r '.metadata.submit_id // empty' <<<"$cur")
    owned=0
    [ "$cur_gen" = "$gen" ] && [ "$cur_sid" = "$submit_id" ] && owned=1

    if [ "$rc" -ne 0 ]; then
        log "error: sbatch for $key failed: $job"
        [ "$owned" -eq 1 ] && claim_update "$key" "$gen" "$cur_mgen" status=queued -submit_id -submitted_at
        continue
    fi
    if [ "$owned" -eq 1 ] && claim_update "$key" "$gen" "$cur_mgen" "slurm_job=$job"; then
        log "submit $key backend=$backend slurm_job=$job"
        inflight[$backend]=$((inflight[$backend] + 1))
        submitted=$((submitted + 1))
    else
        log "warn: $key changed after submit; cancelling slurm job $job"
        scancel "$job"
    fi
done < <(jq -sc 'map(select(.metadata.status == "queued")) | sort_by(.metadata.updated_at | tonumber? // 0) | .[]' <<<"$claims")

log "pass: claims=$(grep -c . <<<"$claims") queued=$queued submitted=$submitted failed=$failed requeued=$requeued inflight_dl=${inflight[DL]} inflight_pt=${inflight[PT]}"
