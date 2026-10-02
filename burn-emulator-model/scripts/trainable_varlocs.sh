#!/bin/bash
set -euo pipefail

# varlocs with complete training data (legalmax outputs_table.csv is written last by ignite)
# for the current data_version, one per line, sorted; run from burn-emulator-model/

data_version=$(grep -oP '^data_version:[[:space:]]*\K\S+' configs/varlocs/current.yaml)
for table in data/training_data/*/"$data_version"/legalmax/outputs_table.csv; do
    [ -f "$table" ] || continue
    basename "$(dirname "$(dirname "$(dirname "$table")")")"
done | LC_ALL=C sort
