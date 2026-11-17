#!/usr/bin/env bash
set -euo pipefail
root=/run/media/sensen/Data2/cell_wound_prototype
env MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  "$root/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python" -B "$@"
