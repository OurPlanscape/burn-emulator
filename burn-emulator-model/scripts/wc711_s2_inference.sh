#!/bin/bash

ARRAY_START=0
ARRAY_END=19

MAX_CONCURRENT=1

ARCHITECTURE=circlepp
VARLOC=WC711
DATA_VERSION=28Aug2026

CONFIG_DIR=configs
DATA_ROOT="data/training_data/${VARLOC}/${DATA_VERSION}"
S2_ROOT="data/outputs_12Aug2026"


export topo_path="${DATA_ROOT}/topo"

run_task () {
    local TASK_ID=$1

    source "$HOME/.bashrc"
    source .venv/bin/activate

    EVAL_CONFIGS=(
        "baseline"
        "legalmax"
    )
    EVAL_CFG=${EVAL_CONFIGS[$(( TASK_ID % 2 ))]}
    IGN=$((TASK_ID / 2))
    N=$((IGN + 1))

    export scenario="${N}_${EVAL_CFG}"
    export ignitions_path="${S2_ROOT}/${N}_${EVAL_CFG}/${N}_ignitions_locations.csv"
    if [ "$EVAL_CFG" = "legalmax" ]; then
        export treatment_fuels_path="${S2_ROOT}/${N}_legalmax"
        export treatment_wind_ang_path="${S2_ROOT}/${N}_baseline/${N}_outputs_table.csv"
    else
        export baseline_fuels_path="${DATA_ROOT}/baseline_FF"
        export baseline_wind_ang_path="${S2_ROOT}/${N}_baseline/${N}_outputs_table.csv"
    fi

    burn_emulator -m evaluate \
        -a "$ARCHITECTURE" \
        -vl "$VARLOC" \
        -dv "$DATA_VERSION" \
        -c "$CONFIG_DIR/${ARCHITECTURE}/model.yaml" \
        -c "$CONFIG_DIR/varlocs/templates/eval_data.yaml"
}

running=0
for TASK_ID in $(seq "$ARRAY_START" "$ARRAY_END"); do
    run_task "$TASK_ID" &
    running=$((running + 1))

    if [ "$running" -ge "$MAX_CONCURRENT" ]; then
        wait -n
        running=$((running - 1))
    fi
done

wait