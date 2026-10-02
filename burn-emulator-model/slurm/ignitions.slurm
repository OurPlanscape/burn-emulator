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
# submit from the repo root: sbatch burn-emulator-model/slurm/ignitions.slurm
# optional: sbatch --export=ALL,VARLOC=<varloc>,NUM_IGNITIONS=<n>,OVERWRITE=1,INPUTS_VERSION=<YYYYMMDD>,IGNITIONS_VERSION=<YYYYMMDD> ...
# (the versions default to current.yaml; make reads them from the environment)
# ray.init() claims every visible cpu, hence --exclusive on the cpu-only node

set -euo pipefail

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID, cpus: $SLURM_CPUS_ON_NODE"

set +u
source "$HOME/.bashrc"
set -u

# sbatch --export=ALL carries the submitting make's MAKEFLAGS/MAKELEVEL into the job
unset MAKEFLAGS MAKELEVEL MFLAGS

MAKE_ARGS=()
[ -n "${NUM_IGNITIONS:-}" ] && MAKE_ARGS+=(NUM_IGNITIONS="$NUM_IGNITIONS")
[ -n "${OVERWRITE:-}" ] && MAKE_ARGS+=(OVERWRITE=1)

if [ -n "${VARLOC:-}" ]; then
    make training-data VARLOC="$VARLOC" ${MAKE_ARGS[@]+"${MAKE_ARGS[@]}"}
else
    make training-data-all ${MAKE_ARGS[@]+"${MAKE_ARGS[@]}"}
fi

echo "Job finished at $(date)"
