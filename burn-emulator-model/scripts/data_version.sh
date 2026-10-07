#!/bin/bash
set -euo pipefail

# prints <inputs_version>_<ignitions_version> from current.yaml, or one key's value

current_yaml="$(dirname "${BASH_SOURCE[0]}")/../configs/varlocs/current.yaml"
key () { grep -oP "^$1:[[:space:]]*\"?\K[0-9]{8}(?=\"?(\s|$))" "$current_yaml" || { echo "error: $1 in $current_yaml must be YYYYMMDD" >&2; exit 1; }; }

if [ $# -gt 0 ]; then
    key "$1"
    exit
fi

inputs=$(key inputs_version)
ignitions=$(key ignitions_version)
echo "${inputs}_${ignitions}"
