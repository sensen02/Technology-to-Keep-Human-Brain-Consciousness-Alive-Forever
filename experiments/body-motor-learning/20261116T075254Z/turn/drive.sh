#!/usr/bin/env bash
P=/run/media/sensen/Data2/cell_wound_prototype
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
name=$2
"$P/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python" -B -u rollout.py "$@" > "${name}.log" 2>&1
code=$?
echo "$code" > "${name}.exit"
exit "$code"
