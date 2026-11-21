#!/usr/bin/env bash
set -euo pipefail

source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
target="${1:?Pass a new absolute output directory}"
[[ "$target" = /* ]]
mkdir -- "$target"
cd -- "$target"

archive=/run/media/sensen/Data2/cell_wound_prototype
neural="$archive/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python"
flight="$archive/experiments/open-brain/20261117T051249Z-bidirectional/flight_env/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 MUJOCO_GL=egl

"$flight" -uB "$source_dir/perching.py" GUIDE_NEAR --case sham --duration .57
"$neural" -uB "$source_dir/flight_neck.py" GUIDE_NEAR NECK_NEAR
"$flight" -uB "$source_dir/perching.py" NEAR --neck NECK_NEAR

"$flight" -uB "$source_dir/perching.py" GUIDE_FAR --case sham --duration .57 --x 8.68 --seed 53
"$neural" -uB "$source_dir/flight_neck.py" GUIDE_FAR NECK_FAR
"$flight" -uB "$source_dir/perching.py" FAR --neck NECK_FAR --x 8.68 --seed 53

"$flight" -uB "$source_dir/perching.py" CUT --neck NECK_NEAR --case cut
"$flight" -uB "$source_dir/perching.py" BLIND --case sham --blank
"$flight" -uB "$source_dir/perching.py" REPLAY --neck NECK_NEAR

"$neural" -uB "$source_dir/intake.py" NEAR MEAL_NEAR
"$neural" -uB "$source_dir/intake.py" FAR MEAL_FAR
"$neural" -uB "$source_dir/intake.py" CUT MEAL_CUT
"$neural" -uB "$source_dir/intake.py" NEAR MEAL_BLOCKED --blocked
"$neural" -uB "$source_dir/verify_flight_fruit.py" --root "$target"
"$flight" -uB "$source_dir/pictures.py" NEAR MEAL_NEAR VISUAL
ffmpeg -hide_banner -loglevel error -xerror -i VISUAL/FLIGHT_FRUIT.mp4 -map 0:v:0 -f null -
