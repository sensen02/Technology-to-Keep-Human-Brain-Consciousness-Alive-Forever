#!/usr/bin/env bash
set -euo pipefail

source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
target="${1:?Pass a new absolute output directory}"
[[ "$target" = /* ]]
mkdir -- "$target"
cp -- "$source_dir/"{flight_manifest,probe_neck,fly,legged_fixture,legged_air,transition,assess,pictures}.py "$target/"
cd -- "$target"

project=/run/media/sensen/Data2/cell_wound_prototype
neural_python="$project/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python"
flight_python="$project/experiments/open-brain/20261117T051249Z-bidirectional/flight_env/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 MUJOCO_GL=egl

"$neural_python" -uB probe_neck.py NECK
"$neural_python" -uB probe_neck.py NECK_REPLAY
"$flight_python" -uB fly.py NECK VALID_17
"$flight_python" -uB fly.py NECK VALID_29 --seed 29
"$flight_python" -uB fly.py NECK VALID_NO_FLUID --cases sham --no-fluid
"$flight_python" -uB fly.py NECK REPLAY_PHASE29 --seed 29 --cases sham left right amp_high
"$flight_python" -uB legged_fixture.py LEGGED_COLLIDABLE
"$flight_python" -uB legged_air.py NECK LEGGED_AIR
"$flight_python" -uB transition.py NECK TRANSITION_A --lift-cm .045 --landing-start-tick 860
"$flight_python" -uB transition.py NECK TRANSITION_B --lift-cm .065 --retraction .55 --landing-start-tick 870 --cases landing_probe landing_cut
"$flight_python" -uB transition.py NECK TRANSITION_C --lift-cm .1 --retraction .55 --landing-start-tick 870 --cases landing_probe landing_cut
"$flight_python" -uB assess.py
"$flight_python" -uB pictures.py
