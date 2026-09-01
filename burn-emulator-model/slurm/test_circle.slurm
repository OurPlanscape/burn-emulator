#!/bin/bash -l
#SBATCH --job-name=test_circle
#SBATCH --output=data/logs/test_circle_%A_%a.out
#SBATCH --error=data/logs/test_circle_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50000
#SBATCH --nodelist=dragon04


echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"

source $HOME/.bashrc
source .venv/bin/activate

CONFIG_DIR=data/configs/varlocs/wc711

echo "Using test config: $TEST_CONFIG"

burn_emulator -m run -c "$CONFIG_DIR/run_smoke.yaml"

echo "Job finished at $(date)"