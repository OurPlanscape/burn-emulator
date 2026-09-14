#!/bin/bash -l

#SBATCH --job-name=train_varlocs
#SBATCH --output=data/logs/train_varlocs_%A_%a.out
#SBATCH --error=data/logs/train_varlocs_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10000
#SBATCH --nodelist=dragon05
#SBATCH --array=0-12%8 # update range to match the number of varlocs in varlocs.txt; throttle (%N) should match SLOTS below for the target node (dragon02=6, dragon04/dragon05=8)
# --cpus-per-task, --mem, --nodelist, --array and SLOTS are meant to be overridden via submit_varlocs.sh

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"

source $HOME/.bashrc
source .venv/bin/activate

CONFIG_DIR=configs
VARLOC_DIR=$CONFIG_DIR/varlocs
VARLOCS_FILE=$VARLOC_DIR/varlocs.txt
CURRENT_FILE=$VARLOC_DIR/current.yaml
TRAIN_TEMPLATE=$VARLOC_DIR/templates/train.yaml

mapfile -t VARLOCS < <(grep -vE '^[[:space:]]*$' "$VARLOCS_FILE")
VARLOC=${VARLOCS[$SLURM_ARRAY_TASK_ID]}

current_key () { grep -oP "^$1:[[:space:]]*\K\S+" "$CURRENT_FILE"; }
ARCHITECTURE=$(current_key architecture)
DATA_VERSION=$(current_key data_version)

if [ -z "$SLOTS" ]; then
    if [ "$SLURMD_NODENAME" = "dragon02" ]; then
        SLOTS=6
    else
        SLOTS=8
    fi
fi
LOCKDIR=/tmp/train_varlocs_gpu_slots
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

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    (
        while true; do
            free -h > data/logs/train_varlocs.out
            nvidia-smi >> data/logs/train_varlocs.out
            echo "=== $(date) ===" >> data/logs/train_varlocs.out
            sleep 60
        done
    ) &
    MONITOR_PID=$!
fi

burn_emulator -m train \
    -a "$ARCHITECTURE" \
    -vl "$VARLOC" \
    -dv "$DATA_VERSION" \
    -c "$CONFIG_DIR/$ARCHITECTURE/model.yaml" \
    -c "$CONFIG_DIR/$ARCHITECTURE/train.yaml" \
    -c "$TRAIN_TEMPLATE"

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    kill $MONITOR_PID 2>/dev/null
fi

echo "Job finished at $(date)"
