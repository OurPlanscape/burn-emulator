#!/bin/bash

ARRAY_START=0
# ARRAY_END=19
ARRAY_END=1
MAX_CONCURRENT=1

run_task () {
    local TASK_ID=$1

    source "$HOME/.bashrc"
    source .venv/bin/activate
    CONFIG_DIR=configs/wc711

    TEST_CONFIGS=(
        "baseline"
        "legalmax"
    )
    TEST_CFG=${TEST_CONFIGS[$(( TASK_ID % 2 ))]}
    IGN=$((TASK_ID / 2))

    burn_emulator -m test \
        -mn "wc711" \
        -c "$CONFIG_DIR/dataloader.yaml" \
        -c "$CONFIG_DIR/circle.yaml" \
        -c "$CONFIG_DIR/test_${TEST_CFG}.yaml" \
        -c "$CONFIG_DIR/s2_${TEST_CFG}/$((IGN+1))_test.yaml"
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