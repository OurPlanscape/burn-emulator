#!/bin/bash
set -euo pipefail

# varlocs with complete training data (legalmax outputs_table.csv) for the current data_version;
# run from burn-emulator-model/

data_version=$(scripts/data_version.sh)
for table in data/training_data/*/"$data_version"/legalmax/outputs_table.csv; do
    [ -f "$table" ] || continue
    basename "$(dirname "$(dirname "$(dirname "$table")")")"
done | LC_ALL=C sort
