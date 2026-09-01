#!/bin/bash -l
#SBATCH --job-name=test_circle_iter
#SBATCH --output=data/logs/test_circle_iter_%A_%a.out
#SBATCH --error=data/logs/test_circle_iter_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10000
#SBATCH --nodelist=dragon04
#SBATCH --array=0-19%8


echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"

source $HOME/.bashrc
source .venv/bin/activate

CONFIG_DIR=data/configs
TEST_CONFIGS=(
    "baseline"
    "legalmax"
)
TEST_CFG=${TEST_CONFIGS[$(( SLURM_ARRAY_TASK_ID % 2 ))]}
IGN=$((SLURM_ARRAY_TASK_ID / 2))

echo "Using model configs - TEST_CFG: $TEST_CFG" for IGN: $((IGN+1))

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    (
        while true; do
            free -h > data/logs/test_circle_iter.out
            nvidia-smi >> data/logs/test_circle_iter.out
            echo "=== $(date) ===" >> data/logs/test_circle_iter.out
            sleep 60
        done
    ) &
    MONITOR_PID=$!
fi

burn_emulator -m test \
    -mn "wc711" \
    -c "$CONFIG_DIR/circle_pp/model.yaml" \
    -c "$CONFIG_DIR/varlocs/wc711/dataloader.yaml" \
    -c "$CONFIG_DIR/varlocs/wc711/test_${TEST_CFG}.yaml" \
    -c "$CONFIG_DIR/varlocs/wc711/s2_${TEST_CFG}/$((IGN+1))_test.yaml"

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    kill $MONITOR_PID 2>/dev/null
fi

echo "Job finished at $(date)"