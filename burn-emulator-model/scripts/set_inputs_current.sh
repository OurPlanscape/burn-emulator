#!/usr/bin/env bash

set -euo pipefail

# points <inputs_root>/current at a hand-made <inputs_root>/<data_version>/ in the bucket's layout;
# publish_inputs.sh uploads it as is
inputs_root="${1:-}"
data_version="${2:-}"

if [[ -z "$inputs_root" || -z "$data_version" ]]; then
    echo "usage: $0 <inputs_root> <data_version>" >&2
    exit 2
fi

# data_version: west fuels date, YYYYMMDD
if [[ ! "$data_version" =~ ^[0-9]{8}$ ]]; then
    echo "error: data_version '$data_version' must be YYYYMMDD" >&2
    exit 1
fi
inputs_dir="${inputs_root%/}/$data_version"

for t in baseline legalmax; do
    for l in cbd cbh cc fbfm th; do
        [[ -f "$inputs_dir/$t/$l.tif" ]] || { echo "error: $inputs_dir/$t/$l.tif not found" >&2; exit 1; }
    done
done
for l in aspect slope_degrees; do
    [[ -f "$inputs_dir/topo/$l.tif" ]] || { echo "error: $inputs_dir/topo/$l.tif not found" >&2; exit 1; }
done
# by hand with the rasters; varlocs/valid_varlocs.gpkg is made by make valid-varlocs
for f in varlocs/all_varlocs.gpkg fbfm/fbfm_behavior_adjectives.csv; do
    [[ -f "$inputs_dir/$f" ]] || { echo "error: $inputs_dir/$f not found" >&2; exit 1; }
done

printf '%s' "$data_version" > "${inputs_root%/}/.current.$$"
mv "${inputs_root%/}/.current.$$" "${inputs_root%/}/current"
echo "current: ${inputs_root%/}/current = $data_version"
