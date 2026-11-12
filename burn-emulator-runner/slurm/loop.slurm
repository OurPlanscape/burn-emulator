#!/bin/bash -l

#SBATCH --job-name=burn-loop
#SBATCH --output=burn-emulator-runner/data/logs/loop/%x_%j.out
#SBATCH --error=burn-emulator-runner/data/logs/loop/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
# runner claim loop (BURN_EMULATOR_LOOP=1) until scancel or the time limit; from the repo root:
# make runner-loop [NODES=<node>] [SBATCH_ARGS=...], or
# sbatch --gres=gpu:1 --exclusive --mem=0 burn-emulator-runner/slurm/loop.slurm
# runs DL and PT claims alike, as each claim's BURN_EMULATOR_BACKEND says; PT sizes its workers to the node
# needs BURN_EMULATOR_OUTPUT_URI and GOOGLE_APPLICATION_CREDENTIALS from
# ~/.config/burn-emulator/env (make env-vars; burn-emulator-model/scripts/load_env.sh);
# inputs and bundles are read under BURN_EMULATOR_{INPUTS,BUNDLES}_ROOT from the same file
# on SIGTERM the current run's claim goes back to queued for another loop

set -euo pipefail

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID, gpus: ${CUDA_VISIBLE_DEVICES:-none}"

set +u
source "$HOME/.bashrc"
source "burn-emulator-runner/${UV_PROJECT_ENVIRONMENT:-.venv-$(uname -m)}/bin/activate"
set -u
source burn-emulator-model/scripts/load_env.sh

for var in BURN_EMULATOR_OUTPUT_URI GOOGLE_APPLICATION_CREDENTIALS; do
    if [ -z "${!var:-}" ]; then
        echo "error: $var is not set (make env-vars)" >&2
        exit 2
    fi
done
if [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "error: GOOGLE_APPLICATION_CREDENTIALS $GOOGLE_APPLICATION_CREDENTIALS not found" >&2
    exit 2
fi

# the Dockerfile's runtime env on GPU (cuda, bfloat16), looping
export PYTHONUNBUFFERED=1 USE_CLOUD_PATHS=1 RUN_DEVICE=cuda RUN_DTYPE=bfloat16 BURN_EMULATOR_LOOP=1
# exec: slurm's SIGTERM reaches the loop directly
exec burn-emulator-runner
