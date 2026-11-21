#!/usr/bin/env bash
set -euo pipefail

source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
target="${1:?Pass a new absolute output directory}"
[[ "$target" = /* ]]
mkdir -- "$target"
cp -- "$source_dir/"{takeoff.py,verify.py,visualize.py,CALIBRATED.json} "$target/"
cd -- "$target"

project=/run/media/sensen/Data2/cell_wound_prototype
parent=/home/sensen/.codex/worktrees/4528/cell_wound_prototype/experiments/open-brain/20261118T072349Z-flight-descending
neural="$project/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python"
flight="$project/experiments/open-brain/20261117T051249Z-bidirectional/flight_env/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 MUJOCO_GL=egl

"$neural" -uB "$parent/probe_neck.py" NECK_FRESH
"$flight" -uB takeoff.py FORMAL17 --neck NECK_FRESH --config CALIBRATED.json --cases takeoff_probe takeoff_cut sham fluid_off legs_only
"$flight" -uB takeoff.py FORMAL29 --neck NECK_FRESH --config CALIBRATED.json --seed 29 --cases takeoff_probe takeoff_cut sham
"$flight" -uB takeoff.py FORMAL53 --neck NECK_FRESH --config CALIBRATED.json --seed 53 --cases takeoff_probe takeoff_cut sham
"$flight" -uB takeoff.py REPLAY29 --neck NECK_FRESH --config CALIBRATED.json --seed 29 --cases takeoff_probe takeoff_cut
"$flight" -uB verify.py
"$flight" -uB visualize.py --video
ffmpeg -hide_banner -loglevel error -xerror -i TAKEOFF.mp4 -map 0:v:0 -f null -
