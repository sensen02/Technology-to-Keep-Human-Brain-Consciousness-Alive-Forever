#!/usr/bin/env bash
set -euo pipefail
task_dir=$(cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(cd -- "$task_dir/../../.." && pwd)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
exec "$project_dir/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python" -uB "$task_dir/$1" "${@:2}"
