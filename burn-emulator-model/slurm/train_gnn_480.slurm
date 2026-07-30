#!/bin/bash -l
#SBATCH --job-name=train_basemax_gnn
#SBATCH --output=/mnt/data/mtruong/burn-emulator/burn-emulator-model/logs/train_basemax_gnn.out
#SBATCH --error=/mnt/data/mtruong/burn-emulator/burn-emulator-model/logs/train_basemax_gnn.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=124000
#SBATCH --nodelist=dragon04



echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"

source $HOME/.bashrc
cd /mnt/data/mtruong/burn-emulator/burn-emulator-model
source /mnt/data/mtruong/burn-emulator/burn-emulator-model/.venv/bin/activate

CONFIG_DIR=/mnt/data/mtruong/burn-emulator/burn-emulator-model/data/configs

(
    while true; do
        free -h > /mnt/data/mtruong/burn-emulator/burn-emulator-model/logs/train_basemax_gnn.out
        nvidia-smi >> /mnt/data/mtruong/burn-emulator/burn-emulator-model/logs/train_basemax_gnn.out
        echo "=== $(date) ===" >> /mnt/data/mtruong/burn-emulator/burn-emulator-model/logs/train_basemax_gnn.out
        sleep 60
    done
) &
MONITOR_PID=$!


burn_emulator -m train \
    -c "$CONFIG_DIR/basemax_480/gnn.yaml" \
    -c "$CONFIG_DIR/basemax_480/train.yaml"

kill $MONITOR_PID 2>/dev/null

echo "Job finished at $(date)"