#!/bin/bash
set -euo pipefail

# add a varloc to the varlocs.txt allow-list once its training has completed;
# locked since concurrent slurm array tasks finish independently. run from burn-emulator-model/

varloc=${1:?usage: $0 <varloc>}
varlocs_txt=configs/varlocs/varlocs.txt

exec {lock_fd}>"$varlocs_txt.lock"
flock "$lock_fd"
if ! grep -qxF "$varloc" "$varlocs_txt"; then
    { grep -vE '^[[:space:]]*$' "$varlocs_txt"; echo "$varloc"; } | LC_ALL=C sort -u > "$varlocs_txt.tmp"
    mv "$varlocs_txt.tmp" "$varlocs_txt"
    echo "added $varloc to $varlocs_txt"
fi
