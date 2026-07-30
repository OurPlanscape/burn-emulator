#!/bin/bash -l
#SBATCH --job-name=test_test
#SBATCH --output=/mnt/data/mtruong/burn-emulator/burn-emulator-model/logs/test_test.out
#SBATCH --error=/mnt/data/mtruong/burn-emulator/burn-emulator-model/logs/test_test.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=31000
#SBATCH --nodelist=dragon05



export SLURM_ARRAY_TASK_ID=0
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"

source $HOME/.bashrc
cd /mnt/data/mtruong/burn-emulator/burn-emulator-model
source /mnt/data/mtruong/burn-emulator/burn-emulator-model/.venv/bin/activate

CONFIG_DIR=/mnt/data/mtruong/burn-emulator/burn-emulator-model/data/configs

MODEL_CONFIGS=(
    "circle"
    "unet"
)

DATA_CONFIGS=(
    "basemax_all"
    "basemax_2400"
    "basemax_1920"
    "basemax_1440"
    "basemax_960"
    "basemax_480"
)

TEST_CONFIGS=(
    "baseline_test"
    "legalmax_test"
)

MID=$(( SLURM_ARRAY_TASK_ID % 2 ))
DID=$(( (SLURM_ARRAY_TASK_ID / 2) % 4 ))
TID=$(( SLURM_ARRAY_TASK_ID / 8 ))

MODEL_CFG=${MODEL_CONFIGS[$MID]}
DATA_CFG=${DATA_CONFIGS[$DID]}
TEST_CFG=${TEST_CONFIGS[$TID]}

echo "Using model config: $MODEL_CFG, data config: $DATA_CFG, test config: $TEST_CFG"

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    (
        while true; do
            free -h > /mnt/data/mtruong/burn-emulator/burn-emulator-model/logs/test_cnns.out
            nvidia-smi >> /mnt/data/mtruong/burn-emulator/burn-emulator-model/logs/test_cnns.out
            echo "=== $(date) ===" >> /mnt/data/mtruong/burn-emulator/burn-emulator-model/logs/test_cnns.out
            sleep 60
        done
    ) &
    MONITOR_PID=$!
fi

burn_emulator -m test \
    -c "$CONFIG_DIR/$DATA_CFG/$MODEL_CFG.yaml" \
    -c "$CONFIG_DIR/$DATA_CFG/$TEST_CFG.yaml" \
    -c "$CONFIG_DIR/$DATA_CFG/test.yaml"

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    kill $MONITOR_PID 2>/dev/null
fi

echo "Job finished at $(date)"