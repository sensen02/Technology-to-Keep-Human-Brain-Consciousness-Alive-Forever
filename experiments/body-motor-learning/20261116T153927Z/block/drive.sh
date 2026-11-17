#!/usr/bin/env bash
set -u
cd "$(dirname "$0")" || exit 2
target=$2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
run_python=/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python
"$run_python" -B rollout.py "$@" > "$target.log" 2>&1
rc=$?
echo "$rc" > "$target.exit"
exit "$rc"
