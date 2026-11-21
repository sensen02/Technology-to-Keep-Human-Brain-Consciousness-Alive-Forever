#!/usr/bin/env bash
set -euo pipefail

source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
target="${1:?Use a new absolute directory next to the original flight-fruit run}"
[[ "$target" = /* && "$(dirname "$target")" = "$(dirname "$source_dir")" ]]
mkdir -- "$target"
cp -- "$source_dir/"{perch_wing.py,inspect_pose.py,verify_stow.py,media_stow.py} "$target/"
cd -- "$target"

archive=/run/media/sensen/Data2/cell_wound_prototype
flight="$archive/experiments/open-brain/20261117T051249Z-bidirectional/flight_env/bin/python"
neural="$archive/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python"
meal="$target/../20261118T234622Z-flight-fruit/intake.py"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 MUJOCO_GL=egl

"$flight" -uB perch_wing.py NEAR
LIBGL_ALWAYS_SOFTWARE=1 "$flight" -uB perch_wing.py FAR --case FAR
"$flight" -uB perch_wing.py REPLAY
"$neural" -uB "$meal" NEAR MEAL_NEAR
"$neural" -uB "$meal" FAR MEAL_FAR
"$flight" -uB verify_stow.py
"$flight" -uB media_stow.py VISUAL
ffmpeg -hide_banner -loglevel error -xerror -i VISUAL/FLIGHT_FRUIT.mp4 -map 0:v:0 -f null -
