#!/bin/bash

MAX_CONCURRENT=2

CONFIG_DIR=configs
VARLOC_DIR="$CONFIG_DIR/varlocs"
VARLOCS_FILE="$VARLOC_DIR/varlocs.txt"
CURRENT_FILE="$VARLOC_DIR/current.yaml"
TRAIN_TEMPLATE="$VARLOC_DIR/templates/train.yaml"

source "$HOME/.bashrc"
source .venv/bin/activate

current_key () { grep -oP "^$1:[[:space:]]*\K\S+" "$CURRENT_FILE"; }
ARCHITECTURE=$(current_key architecture)
DATA_VERSION=$(current_key data_version)

MODEL_YAML="$CONFIG_DIR/$ARCHITECTURE/model.yaml"
TRAIN_YAML="$CONFIG_DIR/$ARCHITECTURE/train.yaml"

mapfile -t VARLOCS < <(grep -vE '^[[:space:]]*(#|$)' "$VARLOCS_FILE")

echo "architecture=$ARCHITECTURE  data_version=$DATA_VERSION  varlocs=${#VARLOCS[@]}"

run_task () {
    local VARLOC=$1
    echo "[train] $VARLOC -> ${VARLOC}_${ARCHITECTURE}_${DATA_VERSION}"
    burn_emulator -m train \
        -a "$ARCHITECTURE" \
        -vl "$VARLOC" \
        -dv "$DATA_VERSION" \
        -c "$MODEL_YAML" \
        -c "$TRAIN_YAML" \
        -c "$TRAIN_TEMPLATE"
}

running=0
for VARLOC in "${VARLOCS[@]}"; do
    run_task "$VARLOC" &
    running=$((running + 1))

    if [ "$running" -ge "$MAX_CONCURRENT" ]; then
        wait -n
        running=$((running - 1))
    fi
done

wait
