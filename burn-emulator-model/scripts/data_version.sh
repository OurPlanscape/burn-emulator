#!/bin/bash
set -euo pipefail

# <inputs_version>_<ignitions_version> from current.yaml: names data/training_data/<varloc>/<data_version>/ and the model;
# with a key (inputs_version | ignitions_version), prints just that one

current_yaml="$(dirname "${BASH_SOURCE[0]}")/../configs/varlocs/current.yaml"
key () { grep -oP "^$1:[[:space:]]*\"?\K[0-9]{8}(?=\"?(\s|$))" "$current_yaml" || { echo "error: $1 in $current_yaml must be YYYYMMDD" >&2; exit 1; }; }

if [ $# -gt 0 ]; then
    key "$1"
    exit
fi

inputs=$(key inputs_version)
ignitions=$(key ignitions_version)
echo "${inputs}_${ignitions}"
