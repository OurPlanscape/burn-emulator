#!/bin/bash -l
set -euo pipefail

# run from the repo root: burn-emulator-model/slurm/submit_ignitions_all.sh [-v <varloc>]... [-n <num_ignitions>] [-o] [-i <inputs_version>] [-g <ignitions_version>]
# one ignitions.slurm job per varloc in the varlocs gpkg (-v restricts to the given varlocs), skipping
# ones whose training data is complete (legalmax outputs_table.csv) unless -o; incomplete ones are
# regenerated from scratch. -i / -g (YYYYMMDD) default to current.yaml

usage () { echo "usage: $0 [-v <varloc>]... [-n <num_ignitions>] [-o] [-i <inputs_version>] [-g <ignitions_version>]" >&2; exit 1; }

SELECTED=()
NUM_IGNITIONS=""
OVERWRITE=""
INPUTS_VERSION=""
IGNITIONS_VERSION=""
while getopts "v:n:oi:g:" opt; do
    case "$opt" in
        v) SELECTED+=("$OPTARG") ;;
        n) NUM_IGNITIONS=$OPTARG ;;
        o) OVERWRITE=1 ;;
        i) INPUTS_VERSION=$OPTARG ;;
        g) IGNITIONS_VERSION=$OPTARG ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))
[ $# -eq 0 ] || usage

MODEL_DIR=burn-emulator-model
DATA_VERSION="${INPUTS_VERSION:-$("$MODEL_DIR"/scripts/data_version.sh inputs_version)}_${IGNITIONS_VERSION:-$("$MODEL_DIR"/scripts/data_version.sh ignitions_version)}"

if [ ${#SELECTED[@]} -gt 0 ]; then
    VARLOCS=("${SELECTED[@]}")
else
    set +u
    source "$MODEL_DIR/${UV_PROJECT_ENVIRONMENT:-.venv-$(uname -m)}/bin/activate"
    set -u
    mapfile -t VARLOCS < <(cd "$MODEL_DIR" && scripts/gpkg_varlocs.sh)
fi
[ ${#VARLOCS[@]} -gt 0 ] || { echo "no varlocs" >&2; exit 1; }

mkdir -p "$MODEL_DIR/data/logs"
echo "data_version=$DATA_VERSION  varlocs=${#VARLOCS[@]}"

for VARLOC in "${VARLOCS[@]}"; do
    if [ -z "$OVERWRITE" ] && [ -f "$MODEL_DIR/data/training_data/$VARLOC/$DATA_VERSION/legalmax/outputs_table.csv" ]; then
        echo "[$VARLOC] skipping: $DATA_VERSION already complete"
        continue
    fi
    echo "[$VARLOC] submitting"
    sbatch \
        --job-name="ignitions_$VARLOC" \
        --export=ALL,VARLOC="$VARLOC",DATA_VERSION="$DATA_VERSION",NUM_IGNITIONS="$NUM_IGNITIONS",OVERWRITE=1 \
        "$(dirname "$0")/ignitions.slurm"
done
