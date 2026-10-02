#!/bin/bash -l

set -euo pipefail

MAX_CONCURRENT=1

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "$MODEL_DIR/${UV_PROJECT_ENVIRONMENT:-.venv-$(uname -m)}/bin/activate"
cd "$MODEL_DIR"

CONFIG_DIR=configs
VARLOC_DIR="$CONFIG_DIR/varlocs"
CURRENT_FILE="$VARLOC_DIR/current.yaml"
TRAIN_TEMPLATE="$VARLOC_DIR/templates/train.yaml"

current_key () { grep -oP "^$1:[[:space:]]*\K\S+" "$CURRENT_FILE"; }
iso8601_date () { [[ "$1" =~ ^[0-9]{8}$ ]] && echo "$1" || date -u -d "$1" +%Y%m%d; }
ARCHITECTURE=$(current_key architecture)
DATA_VERSION=$(current_key data_version)
DATA_VERSION_ISO=$(iso8601_date "$DATA_VERSION") # ISO 8601 for resolve_model_name builds

MODEL_YAML="$CONFIG_DIR/$ARCHITECTURE/model.yaml"
TRAIN_YAML="$CONFIG_DIR/$ARCHITECTURE/train.yaml"

# varlocs with complete training data; each is added to varlocs.txt once its training succeeds
mapfile -t VARLOCS < <(scripts/trainable_varlocs.sh)

echo "architecture=$ARCHITECTURE  data_version=$DATA_VERSION  varlocs=${#VARLOCS[@]}"

run_task () {
    local VARLOC=$1
    echo "[train] $VARLOC -> ${VARLOC}_${ARCHITECTURE}_${DATA_VERSION_ISO}"
    burn_emulator -m train \
        -a "$ARCHITECTURE" \
        -vl "$VARLOC" \
        -dv "$DATA_VERSION" \
        -c "$MODEL_YAML" \
        -c "$TRAIN_YAML" \
        -c "$TRAIN_TEMPLATE"
    scripts/mark_trained.sh "$VARLOC"
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
