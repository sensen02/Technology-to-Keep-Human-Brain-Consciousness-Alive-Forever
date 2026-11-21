#!/usr/bin/env bash
# Render from the Gaussian files plus camera calibration and optical tracking.
set -euo pipefail
package=$(cd "$(dirname "$0")" && pwd)
out=$1
mkdir "$out"
py=${PYTHON:-/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$py" "$package/render.py" "$package/environment.ply" "$package/calibration.json" "$package/estimated.json" "$out" --correct-near --foreground "$package/body"
(cd "$out" && sha256sum -c "$package/EYE_OUTPUT.sha256")
