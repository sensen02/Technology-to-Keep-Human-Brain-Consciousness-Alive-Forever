#!/usr/bin/env bash
set -u
cd "$(dirname -- "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
python=/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python
mode="$1"
run="$2"
shift 2
"$python" -B crossing.py "$mode" "$run" "$@" > "$run.log" 2>&1
status=$?
printf '%s\n' "$status" > "$run.exit"
exit "$status"
