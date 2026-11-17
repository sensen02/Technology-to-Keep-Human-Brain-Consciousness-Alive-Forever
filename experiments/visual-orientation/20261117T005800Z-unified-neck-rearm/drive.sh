#!/usr/bin/env bash
set -eu
cd "$(dirname -- "$0")"
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec ../../banc-walking-recovery/20261110T125712Z/env/bin/python -B "$@"
