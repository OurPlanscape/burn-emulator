#!/bin/bash -l
#SBATCH --job-name=evaluate_circle_iter
#SBATCH --output=data/logs/evaluate_circle_iter_%A_%a.out
#SBATCH --error=data/logs/evaluate_circle_iter_%A_%a.err
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

ARCHITECTURE=circlepp
VARLOC=WC711
DATA_VERSION=28Aug2026

ROOT=data
CONFIG_DIR="$ROOT/configs"
DATA_ROOT="$ROOT/training_data/${VARLOC}/${DATA_VERSION}"
S2_ROOT="$ROOT/outputs_12Aug2026"          # per-ignition scenario outputs

EVAL_CONFIGS=(
    "baseline"
    "legalmax"
)
EVAL_CFG=${EVAL_CONFIGS[$(( SLURM_ARRAY_TASK_ID % 2 ))]}
IGN=$((SLURM_ARRAY_TASK_ID / 2))
N=$((IGN + 1))

echo "Using model configs - EVAL_CFG: $EVAL_CFG for IGN: $N"

export topo_path="${DATA_ROOT}/topo"
export scenario="${N}_${EVAL_CFG}"
export ignitions_path="${S2_ROOT}/${N}_${EVAL_CFG}/${N}_ignitions_locations.csv"
if [ "$EVAL_CFG" = "legalmax" ]; then
    export treatment_fuels_path="${S2_ROOT}/${N}_legalmax"
    export treatment_wind_ang_path="${S2_ROOT}/${N}_baseline/${N}_outputs_table.csv"
else
    export baseline_fuels_path="${DATA_ROOT}/baseline_FF"
    export baseline_wind_ang_path="${S2_ROOT}/${N}_baseline/${N}_outputs_table.csv"
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    (
        while true; do
            free -h > data/logs/evaluate_circle_iter.out
            nvidia-smi >> data/logs/evaluate_circle_iter.out
            echo "=== $(date) ===" >> data/logs/evaluate_circle_iter.out
            sleep 60
        done
    ) &
    MONITOR_PID=$!
fi

burn_emulator -m evaluate \
    -a "$ARCHITECTURE" \
    -vl "$VARLOC" \
    -dv "$DATA_VERSION" \
    -c "$CONFIG_DIR/${ARCHITECTURE}/model.yaml" \
    -c "$CONFIG_DIR/varlocs/templates/eval_data.yaml"

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    kill $MONITOR_PID 2>/dev/null
fi

echo "Job finished at $(date)"
