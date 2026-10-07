#!/bin/bash -l
set -euo pipefail

# run from the repo root: burn-emulator-model/slurm/submit_ignitions_all.sh [-v <varloc>]... [-x <varloc[,varloc...]>]... [-n <num_ignitions>] [-o] [-i <inputs_version>] [-g <ignitions_version>]
# queues gpkg varlocs (-v only these, -x all but these), skipping complete ones unless -o.
# -i / -g default to current.yaml. queue: data/logs/ignitions/ignitions_queue_<stamp>.txt (see queue.sh)

usage () { echo "usage: $0 [-v <varloc>]... [-x <varloc[,varloc...]>]... [-n <num_ignitions>] [-o] [-i <inputs_version>] [-g <ignitions_version>]" >&2; exit 1; }

SELECTED=()
EXCLUDED=()
NUM_IGNITIONS=""
OVERWRITE=""
INPUTS_VERSION=""
IGNITIONS_VERSION=""
while getopts "v:x:n:oi:g:" opt; do
    case "$opt" in
        v) SELECTED+=("$OPTARG") ;;
        x) IFS=", " read -ra PARTS <<<"$OPTARG"; EXCLUDED+=("${PARTS[@]}") ;;
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
if [ ${#EXCLUDED[@]} -gt 0 ]; then
    for varloc in "${EXCLUDED[@]}"; do
        printf '%s\n' "${VARLOCS[@]}" | grep -qxF "$varloc" || echo "warning: excluded $varloc is not in the set to generate" >&2
    done
    mapfile -t VARLOCS < <(printf '%s\n' "${VARLOCS[@]}" | { grep -vxF -f <(printf '%s\n' "${EXCLUDED[@]}") || true; })
fi
[ ${#VARLOCS[@]} -gt 0 ] || { echo "no varlocs" >&2; exit 1; }

SUBMIT=()
for VARLOC in "${VARLOCS[@]}"; do
    if [ -z "$OVERWRITE" ] && [ -f "$MODEL_DIR/data/training_data/$VARLOC/$DATA_VERSION/legalmax/outputs_table.csv" ]; then
        echo "[$VARLOC] skipping: $DATA_VERSION already complete"
        continue
    fi
    SUBMIT+=("$VARLOC")
done
[ ${#SUBMIT[@]} -gt 0 ] || { echo "nothing to submit"; exit 0; }
echo "data_version=$DATA_VERSION  submitting ${#SUBMIT[@]} of ${#VARLOCS[@]} varlocs"

LOG_DIR=$MODEL_DIR/data/logs/ignitions
mkdir -p "$LOG_DIR"
QUEUE=$(realpath "$LOG_DIR")/ignitions_queue_$(date +%Y%m%dT%H%M%S).txt
printf '%s\n' "${SUBMIT[@]}" > "$QUEUE"

JOB_ID=$(sbatch --parsable \
    --output="$LOG_DIR/%x_%A_%a.out" --error="$LOG_DIR/%x_%A_%a.err" \
    --array=0 \
    --export=ALL,VARLOC=,QUEUE="$QUEUE",DATA_VERSION="$DATA_VERSION",NUM_IGNITIONS="$NUM_IGNITIONS",OVERWRITE=1 \
    "$(dirname "$0")/ignitions.slurm")
echo "job ${JOB_ID%%;*}: $QUEUE"
