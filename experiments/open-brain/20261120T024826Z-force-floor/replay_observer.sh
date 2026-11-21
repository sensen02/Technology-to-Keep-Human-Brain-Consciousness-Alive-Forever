#!/usr/bin/env bash
# From the observation package, recompute without any physics assets or scoring data.
set -eu
pkg=$(cd "$(dirname "$0")" && pwd)
out=$1
mkdir "$out"
mkdir "$out/cal" "$out/trial"
cp "$pkg/response.json" "$out/cal/"
cp "$pkg/bottom.npz" "$out/trial/"
cp -r "$pkg/external" "$out/trial/"
py=${PYTHON:-/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python}
OPENBLAS_NUM_THREADS=1 "$py" "$pkg/estimate.py" "$out/trial" "$out/cal" --rig-origin "$pkg/rig_origin.json"
"$py" "$pkg/episodes.py" "$out/trial"
(cd "$out/trial" && sha256sum -c "$pkg/EXPECTED.sha256")
