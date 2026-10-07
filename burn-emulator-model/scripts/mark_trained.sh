#!/bin/bash
set -euo pipefail

# adds <varloc> to data/outputs/varlocs.txt under a lock; extra args run while it is held.
# run from burn-emulator-model/

varloc=${1:?usage: $0 <varloc> [cmd ...]}
varloc=${varloc^^}
shift
varlocs_txt=data/outputs/varlocs.txt

exec {lock_fd}>"$varlocs_txt.lock"
flock "$lock_fd"
touch "$varlocs_txt"
if ! grep -qxF "$varloc" "$varlocs_txt"; then
    { grep -vE '^[[:space:]]*$' "$varlocs_txt" || true; echo "$varloc"; } | LC_ALL=C sort -u > "$varlocs_txt.tmp"
    mv "$varlocs_txt.tmp" "$varlocs_txt"
    echo "added $varloc to $varlocs_txt"
fi

if [ $# -gt 0 ]; then
    "$@"
fi
