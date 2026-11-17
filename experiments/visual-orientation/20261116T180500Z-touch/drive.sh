#!/usr/bin/env bash
set -eu
cd "$(dirname -- "$0")"
env MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ../../banc-walking-recovery/20261110T125712Z/env/bin/python -B "$@"
