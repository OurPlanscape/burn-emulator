# sourced by dispatch.sh, slurm/infer.slurm and make runner-venv
# BURN_EMULATOR_ENV and BURN_EMULATOR_OUTPUTS_URI come from $USER's environment; data/.env holds cluster settings

API_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO_DIR=$(dirname "$API_DIR")
LOG_DIR=$API_DIR/data/logs
umask 002

env_file=$API_DIR/data/.env
if [ ! -f "$env_file" ]; then
    echo "error: $env_file not found; copy burn-emulator-api/.env.example" >&2
    exit 2
fi
set -a
# shellcheck source=/dev/null
source "$env_file"
set +a

: "${INPUTS_ROOT:=$REPO_DIR/burn-emulator-model/data/inputs}"
: "${BUNDLES_ROOT:=$REPO_DIR/burn-emulator-model/data/bundles}"
: "${CLOUDSDK_CONFIG:=$API_DIR/data/gcloud}"
: "${UV_CACHE_DIR:=$API_DIR/data/uv-cache}"

for var in BURN_EMULATOR_ENV BURN_EMULATOR_OUTPUTS_URI KEY_FILE; do
    if [ -z "${!var:-}" ]; then
        echo "error: $var is not set (export it in $USER's login profile; scheduled jobs only see what bash -l loads)" >&2
        exit 2
    fi
done
if [ ! -f "$KEY_FILE" ]; then
    echo "error: KEY_FILE $KEY_FILE not found" >&2
    exit 2
fi
if (( 8#$(stat -c %a "$KEY_FILE") & 8#077 )); then
    echo "error: KEY_FILE $KEY_FILE is readable by group or others; chmod 600 it" >&2
    exit 2
fi

mkdir -p "$CLOUDSDK_CONFIG" "$UV_CACHE_DIR" "$LOG_DIR/infer"
export INPUTS_ROOT BUNDLES_ROOT CLOUDSDK_CONFIG UV_CACHE_DIR
export CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE=$KEY_FILE
export CLOUDSDK_CORE_DISABLE_PROMPTS=1

CLAIMS_URI=${BURN_EMULATOR_OUTPUTS_URI%/}/_claims
RUNNER_BIN=$REPO_DIR/burn-emulator-runner/.venv-$(uname -m)/bin/burn-emulator-runner

log () {
    printf '%s [%s] [%s] [job %s] %s\n' "$(date -u +%FT%TZ)" "$BURN_EMULATOR_ENV" "$USER" "${SLURM_JOB_ID:--}" "$*"
}
