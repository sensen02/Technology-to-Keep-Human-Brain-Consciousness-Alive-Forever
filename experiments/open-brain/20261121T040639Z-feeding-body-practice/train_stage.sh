#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
variant=$1
updates=$2
shift 2
for seed in "$@"; do
    if [[ "$seed" == 88207 || "$seed" == 88209 ]]; then continue; fi
    until [[ -s "aligned-$seed/electrode.npz" ]]; do
        sleep 3
    done
done
nice -n 12 bash env.sh practice.py "$variant" "$@" --updates "$updates"
