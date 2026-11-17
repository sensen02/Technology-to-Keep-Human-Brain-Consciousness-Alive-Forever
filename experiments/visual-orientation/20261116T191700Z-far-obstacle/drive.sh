#!/usr/bin/env bash
set -eu
cd "$(dirname -- "$0")"
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$(pwd)/../20261116T184600Z-approach"
exec ../../banc-walking-recovery/20261110T125712Z/env/bin/python -B "$@"
