#!/usr/bin/env bash
set -euo pipefail
task_dir=$(cd -- "$(dirname -- "$0")" && pwd)
task_project=$(cd -- "$task_dir/../../.." && pwd)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl MPLBACKEND=Agg
exec "$task_project/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python" -uB "$task_dir/$1" "${@:2}"
