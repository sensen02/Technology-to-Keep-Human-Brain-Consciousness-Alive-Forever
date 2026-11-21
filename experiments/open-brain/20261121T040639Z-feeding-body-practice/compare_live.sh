#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
variant=$1
shift
while [[ ! -s "$variant/training.json" ]]; do sleep 5; done
for seed in "$@"; do
    if [[ "$seed" != 88207 && "$seed" != 88209 && "$seed" != 88211 ]]; then
        while [[ ! -s "aligned-$seed/electrode.npz" ]]; do sleep 5; done
    fi
    for candidate in previous "$variant"; do
        if [[ ! -s "live-$seed-on-$candidate/result.json" ]]; then
            bash run_live_capped.sh "$seed" on 6.4 "$candidate" > "logs/live-$seed-$candidate.log" 2>&1
        fi
    done
done
