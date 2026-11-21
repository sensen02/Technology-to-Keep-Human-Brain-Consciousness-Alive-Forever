#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
archive=/run/media/sensen/Data2/cell_wound_prototype
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MUJOCO_GL=egl TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 CUDA_VISIBLE_DEVICES=-1 LP_NUM_THREADS=2
exec "$archive/experiments/open-brain/20261117T051249Z-bidirectional/flight_env/bin/python" -uB "$@"
