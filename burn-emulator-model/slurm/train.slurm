#!/bin/bash -l

#SBATCH --job-name=train
#SBATCH --output=burn-emulator-model/data/logs/%x_%j.out
#SBATCH --error=burn-emulator-model/data/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10000
#SBATCH --nodelist=dragon05
# trains one varloc: sbatch --export=ALL,VARLOC=<varloc> burn-emulator-model/slurm/train.slurm
# --job-name, --cpus-per-task, --mem, --nodelist, SLOTS, PUBLISH and MONITOR are meant to be overridden via submit_train_all.sh
# PUBLISH=1 bundles and publishes the varloc after a successful train, then publishes the varlocs txt + gpkg
# (needs BURN_EMULATOR_MODELS_URI and BURN_EMULATOR_INPUTS_URI)

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID, varloc: ${VARLOC:-}"

source $HOME/.bashrc
source "burn-emulator-model/${UV_PROJECT_ENVIRONMENT:-.venv-$(uname -m)}/bin/activate"
cd burn-emulator-model

CONFIG_DIR=configs
VARLOC_DIR=$CONFIG_DIR/varlocs
CURRENT_FILE=$VARLOC_DIR/current.yaml
TRAIN_TEMPLATE=$VARLOC_DIR/templates/train.yaml

[ -n "${VARLOC:-}" ] || { echo "error: pass VARLOC via sbatch --export=ALL,VARLOC=<varloc>" >&2; exit 2; }

if [ -n "${PUBLISH:-}" ]; then
    for var in BURN_EMULATOR_MODELS_URI BURN_EMULATOR_INPUTS_URI; do
        [ -n "${!var:-}" ] || { echo "error: PUBLISH=1 needs $var exported" >&2; exit 2; }
    done
fi

current_key () { grep -oP "^$1:[[:space:]]*\"?\K[^\"[:space:]]+" "$CURRENT_FILE"; }
ARCHITECTURE=$(current_key architecture)
DATA_VERSION=$(scripts/data_version.sh)

if [ -z "${SLOTS:-}" ]; then
    if [ "$SLURMD_NODENAME" = "dragon02" ]; then
        SLOTS=6
    else
        SLOTS=8
    fi
fi
LOCKDIR=/tmp/train_gpu_slots
mkdir -p "$LOCKDIR"

SLOT=""
while [ -z "$SLOT" ]; do
    for i in $(seq 0 $((SLOTS - 1))); do
        exec {slot_fd}>"$LOCKDIR/${SLURMD_NODENAME}.slot${i}"
        if flock -n "$slot_fd"; then
            SLOT=$i
            break
        fi
        exec {slot_fd}>&-
    done
    [ -z "$SLOT" ] && { echo "all $SLOTS slots busy on $SLURMD_NODENAME, waiting..."; sleep 15; }
done

if [ "$SLURMD_NODENAME" = "dragon02" ]; then
    export CUDA_VISIBLE_DEVICES=$SLOT
else
    export CUDA_VISIBLE_DEVICES=0
fi

echo "[gpu] $SLURMD_NODENAME slot $SLOT -> CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[train] $VARLOC -> ${VARLOC}_${ARCHITECTURE}_${DATA_VERSION}"

if [ -n "${MONITOR:-}" ]; then
    (
        while true; do
            free -h > data/logs/train.out
            nvidia-smi >> data/logs/train.out
            echo "=== $(date) ===" >> data/logs/train.out
            sleep 60
        done
    ) &
    MONITOR_PID=$!
fi

# sbatch --export=ALL carries the submitting make's MAKEFLAGS/MAKELEVEL into the job
unset MAKEFLAGS MAKELEVEL MFLAGS

status=0
burn_emulator -m train \
    -a "$ARCHITECTURE" \
    -vl "$VARLOC" \
    -dv "$DATA_VERSION" \
    -c "$CONFIG_DIR/$ARCHITECTURE/model.yaml" \
    -c "$CONFIG_DIR/$ARCHITECTURE/train.yaml" \
    -c "$TRAIN_TEMPLATE" || status=$?
if [ "$status" -eq 0 ] && [ -n "${PUBLISH:-}" ]; then
    echo "[bundle] $VARLOC"
    make -C .. model-bundle VARLOC="$VARLOC" || status=$?
fi
if [ "$status" -eq 0 ] && [ -n "${PUBLISH:-}" ]; then
    echo "[publish] $VARLOC"
    make -C .. publish-model VARLOC="$VARLOC" BUNDLE_DIR= || status=$?
fi
# with PUBLISH the varloc joins the allow-list only after its model is live; the txt update,
# gpkg rebuild and upload share one lock so concurrent finishes each publish the full list
if [ "$status" -eq 0 ] && [ -n "${PUBLISH:-}" ]; then
    echo "[publish-varlocs] $VARLOC"
    scripts/mark_trained.sh "$VARLOC" make -C .. publish-varlocs DATA_VERSION= || status=$?
elif [ "$status" -eq 0 ]; then
    scripts/mark_trained.sh "$VARLOC"
fi

if [ -n "${MONITOR:-}" ]; then
    kill $MONITOR_PID 2>/dev/null
fi

echo "Job finished at $(date)"
exit "$status"
