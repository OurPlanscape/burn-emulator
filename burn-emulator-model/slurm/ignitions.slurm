#!/bin/bash -l

#SBATCH --job-name=ignitions
#SBATCH --output=burn-emulator-model/data/logs/ignitions/%x_%j.out
#SBATCH --error=burn-emulator-model/data/logs/ignitions/%x_%j.err
#SBATCH --partition=scratch
#SBATCH --nodelist=dragon03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
# generates training data for one varloc:
# sbatch --export=ALL,VARLOC=<varloc>[,DATA_VERSION=<inputs>_<ignitions>,NUM_IGNITIONS=<n>,OVERWRITE=1] burn-emulator-model/slurm/ignitions.slurm
# or QUEUE=<file> (from submit_ignitions_all.sh): generates varlocs from the queue until none are pending (see queue.sh)
# DATA_VERSION defaults to current.yaml
# ray.init() claims every visible cpu, hence --exclusive on the cpu-only node

set -euo pipefail

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID, Array: ${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}, cpus: $SLURM_CPUS_ON_NODE"

set +u
source "$HOME/.bashrc"
source "burn-emulator-model/${UV_PROJECT_ENVIRONMENT:-.venv-$(uname -m)}/bin/activate"
set -u
cd burn-emulator-model
mkdir -p data/logs/ignitions
source slurm/queue.sh

if [ -n "${QUEUE:-}" ]; then
    [ -f "$QUEUE" ] || { echo "error: queue file $QUEUE not found" >&2; exit 2; }
else
    [ -n "${VARLOC:-}" ] || { echo "error: pass VARLOC via sbatch --export=ALL,VARLOC=<varloc>" >&2; exit 2; }
fi
DATA_VERSION=${DATA_VERSION:-$(scripts/data_version.sh)}

run_varloc () {
    local ARGS=(-m ignite -vl "$1" -dv "$DATA_VERSION")
    [ -n "${NUM_IGNITIONS:-}" ] && ARGS+=(-ni "$NUM_IGNITIONS")
    [ -n "${OVERWRITE:-}" ] && ARGS+=(-ow)
    echo "[ignite] $1 -> data/training_data/$1/$DATA_VERSION"
    burn_emulator "${ARGS[@]}"
}

status=0
if [ -n "${QUEUE:-}" ]; then
    queue_run ignitions/ignitions run_varloc || status=$?
else
    run_varloc "$VARLOC" || status=$?
fi

echo "Job finished at $(date)"
exit "$status"
