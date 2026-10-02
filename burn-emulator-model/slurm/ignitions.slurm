#!/bin/bash -l

#SBATCH --job-name=ignitions
#SBATCH --output=burn-emulator-model/data/logs/%x_%j.out
#SBATCH --error=burn-emulator-model/data/logs/%x_%j.err
#SBATCH --partition=scratch
#SBATCH --nodelist=dragon03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
# generates training data for one varloc:
# sbatch --export=ALL,VARLOC=<varloc>[,DATA_VERSION=<inputs>_<ignitions>,NUM_IGNITIONS=<n>,OVERWRITE=1] burn-emulator-model/slurm/ignitions.slurm
# or, as an array task from submit_ignitions_all.sh, line SLURM_ARRAY_TASK_ID of VARLOCS_LIST; DATA_VERSION defaults to current.yaml
# ray.init() claims every visible cpu, hence --exclusive on the cpu-only node

set -euo pipefail

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID, Array Task ID: ${SLURM_ARRAY_TASK_ID:-}, cpus: $SLURM_CPUS_ON_NODE"

set +u
source "$HOME/.bashrc"
source "burn-emulator-model/${UV_PROJECT_ENVIRONMENT:-.venv-$(uname -m)}/bin/activate"
set -u
cd burn-emulator-model

if [ -z "${VARLOC:-}" ] && [ -n "${VARLOCS_LIST:-}" ]; then
    mapfile -t VARLOCS < "$VARLOCS_LIST"
    VARLOC=${VARLOCS[$SLURM_ARRAY_TASK_ID]}
fi
[ -n "${VARLOC:-}" ] || { echo "error: pass VARLOC via sbatch --export=ALL,VARLOC=<varloc>" >&2; exit 2; }
DATA_VERSION=${DATA_VERSION:-$(scripts/data_version.sh)}

ARGS=(-m ignite -vl "$VARLOC" -dv "$DATA_VERSION")
[ -n "${NUM_IGNITIONS:-}" ] && ARGS+=(-ni "$NUM_IGNITIONS")
[ -n "${OVERWRITE:-}" ] && ARGS+=(-ow)

echo "[ignite] $VARLOC -> data/training_data/$VARLOC/$DATA_VERSION"
burn_emulator "${ARGS[@]}"

echo "Job finished at $(date)"
