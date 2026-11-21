#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runner="$root/../20261119T012530Z-survival-orchard/run_env.sh"
out=$(realpath -m -- "${1:?provide an unused output directory}")
mkdir -- "$out"
"$runner" "$root/compile_solved.py"
"$runner" "$root/compile_measured.py"
"$runner" "$root/slope_solved.py" "$out/SOLVED" --angles 0 -10 10 -20 20
"$runner" "$root/contact_walk.py" "$out/CONTACT_FRAME" --angles 20 -20 10 -10 0
"$runner" "$root/support_coordinates.py" "$out/TANGENT" --angles 20 -20 10 -10 0
"$runner" "$root/adhesion_probe.py" "$out/LOW_ADHESION" --angles 20 -20 --limit .35
"$runner" "$root/height_probe.py" "$out/CLEARANCE" --angles 20 --clearance-cm .015
"$runner" "$root/support_coordinates.py" "$out/TANGENT_REPEAT" --angles -20 20
"$runner" "$root/support_coordinates.py" "$out/TANGENT_HELDOUT" --angles -20 20 --yaws 8
"$runner" "$root/sensor_only.py" "$out/SENSOR_ONLY" --angles -20 20
"$runner" "$root/audit.py" "$out" --groups SOLVED CONTACT_FRAME TANGENT LOW_ADHESION CLEARANCE TANGENT_REPEAT TANGENT_HELDOUT SENSOR_ONLY --render
