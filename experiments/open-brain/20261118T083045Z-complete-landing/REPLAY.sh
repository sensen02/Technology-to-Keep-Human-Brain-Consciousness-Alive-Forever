#!/usr/bin/env bash
set -euo pipefail

source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
target="${1:?Pass a new absolute output directory}"
[[ "$target" = /* ]]
mkdir -- "$target"
cp -- "$source_dir/"{neck.py,land.py,verify_landing.py,pictures_landing.py,CALIBRATED.json} "$target/"
cd -- "$target"

project=/run/media/sensen/Data2/cell_wound_prototype
neural="$project/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python"
flight="$project/experiments/open-brain/20261117T051249Z-bidirectional/flight_env/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 MUJOCO_GL=egl

"$neural" -uB neck.py NECK
for seed in 17 29 53; do
  "$flight" -uB land.py NECK "FORMAL$seed" --seed "$seed" --config CALIBRATED.json
done
"$flight" -uB land.py NECK REPLAY29 --seed 29 --config CALIBRATED.json --cases intact cut
"$flight" -uB verify_landing.py
"$flight" -uB pictures_landing.py FORMAL17 VISUAL
ffmpeg -hide_banner -loglevel error -xerror -i VISUAL/LANDING.mp4 -map 0:v:0 -f null -
