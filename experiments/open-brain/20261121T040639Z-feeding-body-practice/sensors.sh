#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for id in "$@"; do
    folder="aligned-$id"
    while [[ ! -f "$folder/floor.npz" ]]; do
        sleep 3
        [[ ! -f logs/collection.exit || $(cat logs/collection.exit) == 0 ]] || exit 1
    done
    if [[ ! -f "$folder/external.npz" ]]; then
        OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 12 /run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261117T051249Z-bidirectional/flight_env/bin/python -uB prepare.py "$folder"
    fi
    if [[ ! -f "$folder/electrode.npz" ]]; then
        nice -n 12 bash env.sh voltages.py "$folder"
    fi
done
