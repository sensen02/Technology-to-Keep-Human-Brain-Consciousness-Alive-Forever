#!/usr/bin/env bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
P=/run/media/sensen/Data2/cell_wound_prototype
id=$2
"$P/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python" -B -u run_restart.py "$@" > "${id}.log" 2>&1
code=$?
printf '%s\n' "$code" > "${id}.exit"
exit "$code"
