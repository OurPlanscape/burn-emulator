#!/bin/bash -l

#SBATCH --job-name=train
#SBATCH --output=burn-emulator-model/data/logs/%x_%j.out
#SBATCH --error=burn-emulator-model/data/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10000
# one varloc: sbatch --nodelist=<node> [--gres=gpu:1] --export=ALL,VARLOC=<varloc> burn-emulator-model/slurm/train.slurm
# QUEUE=<file> (from submit_train_all.sh): trains varlocs from the queue until it is empty
# PUBLISH=1 also bundles + publishes (needs BURN_EMULATOR_MODELS_URI and BURN_EMULATOR_INPUTS_URI)

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID, Array Task ID: ${SLURM_ARRAY_TASK_ID:-}"

source $HOME/.bashrc
source "burn-emulator-model/${UV_PROJECT_ENVIRONMENT:-.venv-$(uname -m)}/bin/activate"
cd burn-emulator-model

CONFIG_DIR=configs
VARLOC_DIR=$CONFIG_DIR/varlocs
CURRENT_FILE=$VARLOC_DIR/current.yaml
TRAIN_TEMPLATE=$VARLOC_DIR/templates/train.yaml

if [ -n "${QUEUE:-}" ]; then
    [ -f "$QUEUE" ] || { echo "error: queue file $QUEUE not found" >&2; exit 2; }
    [ -n "${SLURM_ARRAY_TASK_ID:-}" ] || { echo "error: QUEUE needs an array task (SLURM_ARRAY_TASK_ID)" >&2; exit 2; }
    CLAIM_DIR=${QUEUE%.txt}.d
    STATUS_FILE=${QUEUE%.txt}.status
    mkdir -p "$CLAIM_DIR"
else
    [ -n "${VARLOC:-}" ] || { echo "error: pass VARLOC via sbatch --export=ALL,VARLOC=<varloc>" >&2; exit 2; }
fi

if [ -n "${PUBLISH:-}" ]; then
    for var in BURN_EMULATOR_MODELS_URI BURN_EMULATOR_INPUTS_URI; do
        [ -n "${!var:-}" ] || { echo "error: PUBLISH=1 needs $var exported" >&2; exit 2; }
    done
fi

current_key () { grep -oP "^$1:[[:space:]]*\"?\K[^\"[:space:]]+" "$CURRENT_FILE"; }
ARCHITECTURE=$(current_key architecture)
DATA_VERSION=$(scripts/data_version.sh)

# GPU: slurm-assigned (--gres), else array task id (queue) or a free /tmp flock slot
if [ -n "${SLURM_JOB_GPUS:-}" ]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$SLURM_JOB_GPUS}
    echo "[gpu] $SLURMD_NODENAME slurm-assigned GPU $SLURM_JOB_GPUS -> CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
else
    if [ -n "${QUEUE:-}" ]; then
        SLOT=$SLURM_ARRAY_TASK_ID
    else
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
    fi

    if [ "$SLURMD_NODENAME" = "dragon02" ]; then
        export CUDA_VISIBLE_DEVICES=$SLOT
    else
        export CUDA_VISIBLE_DEVICES=0
    fi
    echo "[gpu] $SLURMD_NODENAME slot $SLOT -> CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

[ "${SLURM_ARRAY_TASK_ID:-0}" = 0 ] || MONITOR=""
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

# train -> bundle -> publish -> mark_trained for one varloc; returns the first failing step's status
run_varloc () {
    local VARLOC=$1
    local status=0
    echo "[train] $VARLOC -> ${VARLOC}_${ARCHITECTURE}_${DATA_VERSION}"
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
    return "$status"
}

status=0
if [ -n "${QUEUE:-}" ]; then
    # read the queue up front so nothing in run_varloc can consume it from stdin
    mapfile -t QUEUED < "$QUEUE"
    for varloc in "${QUEUED[@]}"; do
        [ -n "$varloc" ] || continue
        mkdir "$CLAIM_DIR/$varloc" 2>/dev/null || continue
        echo "[queue] $varloc claimed at $(date)"
        result=ok
        run_varloc "$varloc" \
            > "data/logs/train_${varloc}_${SLURM_JOB_ID}.out" \
            2> "data/logs/train_${varloc}_${SLURM_JOB_ID}.err" || { result=failed; status=1; }
        ( flock 9; echo "$varloc $result $SLURM_JOB_ID" >> "$STATUS_FILE" ) 9>>"$STATUS_FILE"
        echo "[queue] $varloc $result at $(date)"
    done
else
    run_varloc "$VARLOC" || status=$?
fi

if [ -n "${MONITOR:-}" ]; then
    kill $MONITOR_PID 2>/dev/null
fi

echo "Job finished at $(date)"
exit "$status"
