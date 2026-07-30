#!/bin/bash -l
#SBATCH --job-name=test_circle_iter
#SBATCH --output=/mnt/data/mtruong/burn-emulator/burn-emulator-model/logs/test_circle_iter_%A_%a.out
#SBATCH --error=/mnt/data/mtruong/burn-emulator/burn-emulator-model/logs/test_circle_iter_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120000
#SBATCH --nodelist=dragon04
#SBATCH --array=0-1%1


echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"

source $HOME/.bashrc
cd /mnt/data/mtruong/burn-emulator/burn-emulator-model
source .venv/bin/activate

CONFIG_DIR=/mnt/data/mtruong/burn-emulator/burn-emulator-model/configs/basemax_480

TEST_CONFIGS=(
    "baseline"
    "legalmax"
)

TEST_CFG=${TEST_CONFIGS[$SLURM_ARRAY_TASK_ID]}

echo "Using model config: test config: $TEST_CFG"

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    (
        while true; do
            free -h > /mnt/data/mtruong/burn-emulator/burn-emulator-model/logs/test_circle_iter.out
            nvidia-smi >> /mnt/data/mtruong/burn-emulator/burn-emulator-model/logs/test_circle_iter.out
            echo "=== $(date) ===" >> /mnt/data/mtruong/burn-emulator/burn-emulator-model/logs/test_circle_iter.out
            sleep 60
        done
    ) &
    MONITOR_PID=$!
fi

burn_emulator -m test_iter \
    -c "$CONFIG_DIR/circle.yaml" \
    -c "$CONFIG_DIR/test.yaml" \
    -c "$CONFIG_DIR/iters2_$TEST_CFG.yaml"

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    kill $MONITOR_PID 2>/dev/null
fi

echo "Job finished at $(date)"