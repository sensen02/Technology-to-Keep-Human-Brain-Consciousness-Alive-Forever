#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=$1
mkdir "$out"
cp "$here/rig.json" "$here/observations.json" "$out/"
cp -r "$here/external" "$out/"
py=${PYTHON:-/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$py" "$here/solve.py" "$out"
(cd "$out" && sha256sum -c "$here/POSE.sha256")
