#!/bin/bash -l

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "usage: $0 <varloc> <outputs_root>" >&2
    exit 1
fi

VARLOC=$1
OUTPUTS_ROOT=$(realpath "$2")

MAX_CONCURRENT=1

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "$MODEL_DIR/.venv/bin/activate"
cd "$MODEL_DIR"

CONFIG_DIR=configs
VARLOC_DIR="$CONFIG_DIR/varlocs"
CURRENT_FILE="$VARLOC_DIR/current.yaml"
EVAL_TEMPLATE="$VARLOC_DIR/templates/eval_data.yaml"

current_key () { grep -oP "^$1:[[:space:]]*\K\S+" "$CURRENT_FILE"; }
ARCHITECTURE=$(current_key architecture)
DATA_VERSION=$(current_key data_version)

DATA_ROOT="data/training_data/${VARLOC}/${DATA_VERSION}"
MODEL_YAML="$CONFIG_DIR/${ARCHITECTURE}/model.yaml"

export topo_path="${DATA_ROOT}/topo"

echo "architecture=$ARCHITECTURE  data_version=$DATA_VERSION  varloc=$VARLOC"

run_task () {
    local SCENARIO_DIR=$1
    local N=$2
    local EVAL_CFG=$3

    source "$MODEL_DIR/.venv/bin/activate"

    local BASELINE_DIR="${SCENARIO_DIR}/${N}_baseline"

    export scenario="$(basename "$SCENARIO_DIR")_${EVAL_CFG}"
    # ignitions_path stays a *_ignitions_locations.csv file so VarLoc skips the
    # geojson/gpkg windowing path and reads fuels at their full raster extent
    export ignitions_path="${SCENARIO_DIR}/${N}_${EVAL_CFG}/${N}_ignitions_locations.csv"
    if [ "$EVAL_CFG" = "legalmax" ]; then
        export treatment_fuels_path="${SCENARIO_DIR}/${N}_legalmax"
        export treatment_wind_ang_path="${BASELINE_DIR}/${N}_outputs_table.csv"
    else
        export baseline_fuels_path="${DATA_ROOT}/baseline_FF"
        export baseline_wind_ang_path="${BASELINE_DIR}/${N}_outputs_table.csv"
    fi

    burn_emulator -m evaluate \
        -a "$ARCHITECTURE" \
        -vl "$VARLOC" \
        -dv "$DATA_VERSION" \
        -c "$MODEL_YAML" \
        -c "$EVAL_TEMPLATE"
}

mapfile -t SCENARIO_DIRS < <(find "$OUTPUTS_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
EVAL_CONFIGS=(baseline legalmax)

running=0
for SCENARIO_DIR in "${SCENARIO_DIRS[@]}"; do
    mapfile -t NS < <(find "$SCENARIO_DIR" -maxdepth 1 -type d -name '*_baseline' -printf '%f\n' \
        | sed 's/_baseline$//' | sort -n)

    for N in "${NS[@]}"; do
        for EVAL_CFG in "${EVAL_CONFIGS[@]}"; do
            run_task "$SCENARIO_DIR" "$N" "$EVAL_CFG" &
            running=$((running + 1))

            if [ "$running" -ge "$MAX_CONCURRENT" ]; then
                wait -n
                running=$((running - 1))
            fi
        done
    done
done

wait
