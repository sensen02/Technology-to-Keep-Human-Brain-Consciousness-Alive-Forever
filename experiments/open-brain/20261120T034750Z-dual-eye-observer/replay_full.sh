#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
prev=/home/sensen/.codex/worktrees/a444/cell_wound_prototype/experiments/open-brain/20261120T024826Z-force-floor
source "$prev/setup.sh"
mkdir "$1"
out=$(cd "$1" && pwd)
cp "$here/rig.json" "$out/"
for a in surface fiducials; do ln -s "$prev/$a.pc2" "$out/$a.pc2"; done
"$bl" -b "$prev/walking_sensor.blend" --python-exit-code 1 -P "$here/capture.py" -- "$src/input.json" "$out"
"$py" "$here/solve.py" "$out"
"$bl" -b "$out/tagged_head.blend" --python-exit-code 1 -P "$here/render_eyes.py" -- "$out"
"$py" "$here/readout.py" "$out" estimated.json "$out/compound.npz"
EYE_REFERENCE=1 "$bl" -b "$out/tagged_head.blend" --python-exit-code 1 -P "$here/render_eyes.py" -- "$out"
"$py" "$here/readout.py" "$out" reference.json "$out/compound_reference.npz"
"$py" "$here/audit.py" "$out"
"$py" "$here/figures.py" "$out"
"$bl" -b "$out/reconstructed_eyes.blend" --python-exit-code 1 -P "$here/check_scene.py" -- "$out"
