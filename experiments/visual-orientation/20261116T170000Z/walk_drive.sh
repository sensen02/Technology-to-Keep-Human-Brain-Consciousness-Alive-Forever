#!/usr/bin/env bash
set -eu
cd "$(dirname -- "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
exec ../../../experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B walk_pilot.py "$@"
