#!/bin/bash -l
#SBATCH --job-name=train_test
#SBATCH --output=/mnt/data/mtruong/burn-emulator/burn-emulator-model/logs/train_circle_%A_%a.out
#SBATCH --error=/mnt/data/mtruong/burn-emulator/burn-emulator-model/logs/train_circle_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50000
#SBATCH --nodelist=dragon04
#SBATCH --array=0-1%2



echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"

source $HOME/.bashrc
cd /mnt/data/mtruong/burn-emulator/burn-emulator-model
source /mnt/data/mtruong/burn-emulator/burn-emulator-model/.venv/bin/activate

CONFIG_DIR=/mnt/data/mtruong/burn-emulator/burn-emulator-model/configs/wc711

TRAIN_CONFIGS=(
    "train_of"
    "train_sm"
)

TRAIN_CFG=${TRAIN_CONFIGS[$SLURM_ARRAY_TASK_ID]}

echo "Using train config: $TRAIN_CFG"

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    (
        while true; do
            free -h > /mnt/data/mtruong/burn-emulator/burn-emulator-model/logs/train_circle.out
            nvidia-smi >> /mnt/data/mtruong/burn-emulator/burn-emulator-model/logs/train_circle.out
            echo "=== $(date) ===" >> /mnt/data/mtruong/burn-emulator/burn-emulator-model/logs/train_circle.out
            sleep 60
        done
    ) &
    MONITOR_PID=$!
fi

burn_emulator -m train \
    -c "$CONFIG_DIR/$TRAIN_CFG.yaml" \
    -c "$CONFIG_DIR/dataloader.yaml" \
    -c "$CONFIG_DIR/dataset.yaml" \
    -c "$CONFIG_DIR/circle.yaml"

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    kill $MONITOR_PID 2>/dev/null
fi

echo "Job finished at $(date)"