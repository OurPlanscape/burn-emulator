#!/bin/bash -l
# one inference run, submitted by scripts/dispatch.sh with
#   --export=NONE,API_DIR=<api dir>,CLAIM=<job id>,CLAIM_GEN=<API claim generation>,SUBMIT_ID=<uuid>,BACKEND=DL|PT
# takes the claim (submitted -> running), then execs the runner (the Cloud Run image's entrypoint) with the env the API stored in the claim,
# /inputs and /models rewritten to INPUTS_ROOT and BUNDLES_ROOT. Resources come from the sbatch args (SLURM_<BACKEND>_ARGS).
set -uo pipefail

source "$API_DIR/scripts/env.sh"
source "$API_DIR/scripts/claims.sh"

cur=$(claim_get "$CLAIM") || { log "claim $CLAIM gone; exiting"; exit 0; }
gen=$(jq -r .generation <<<"$cur")
mgen=$(jq -r .metageneration <<<"$cur")
status=$(jq -r '.metadata.status // empty' <<<"$cur")
sid=$(jq -r '.metadata.submit_id // empty' <<<"$cur")
if [ "$gen" != "$CLAIM_GEN" ] || [ "$status" != submitted ] || [ "$sid" != "$SUBMIT_ID" ]; then
    log "claim $CLAIM superseded (generation $gen, status $status); exiting"
    exit 0
fi
if ! claim_update "$CLAIM" "$CLAIM_GEN" "$mgen" status=running "updated_at=$(date +%s)"; then
    log "claim $CLAIM changed before takeover; exiting"
    exit 0
fi

body=$(claim_body "$CLAIM" "$CLAIM_GEN") || { log "error: reading body of $CLAIM failed"; exit 1; }
mapfile -t vars < <(jq -r 'to_entries[] | "\(.key)=\(.value)"' <<<"$body" \
    | sed -e "s#=/inputs/#=$INPUTS_ROOT/#" -e "s#=/models/#=$BUNDLES_ROOT/#")

case $BACKEND in
    DL) device=cuda ;;
    *) device=cpu ;;
esac
vars+=(
    "BURN_EMULATOR_CLAIM_GENERATION=$CLAIM_GEN"
    "USE_CLOUD_PATHS=1"
    "RUN_DTYPE=bfloat16"
    "RUN_DEVICE=$device"
    "GOOGLE_APPLICATION_CREDENTIALS=$KEY_FILE"
)
for pass in HOME PATH USER TMPDIR LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES SLURM_JOB_ID SLURM_CPUS_PER_TASK \
    HTTPS_PROXY HTTP_PROXY NO_PROXY https_proxy http_proxy no_proxy; do
    [ -n "${!pass:-}" ] && vars+=("$pass=${!pass}")
done

if [ ! -x "$RUNNER_BIN" ]; then
    log "error: $RUNNER_BIN missing; run make runner-venv"
    exit 1
fi
log "run $CLAIM backend=$BACKEND"
exec env -i "${vars[@]}" "$RUNNER_BIN"
