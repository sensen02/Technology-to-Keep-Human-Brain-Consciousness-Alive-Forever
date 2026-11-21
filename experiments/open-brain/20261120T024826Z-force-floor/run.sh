#!/usr/bin/env bash
# Rebuild in a fresh independent directory.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
mkdir "$1"
out=$(cd "$1" && pwd)
cp "$here"/*.py "$here/setup.sh" "$out/"
cd "$out"
source setup.sh
"$bl" -b "$src/marked_fly_small_chamber.blend" --python-exit-code 1 -P build_insert.py -- "$PWD" > build.log
"$bl" -b "$src/marked_fly_small_chamber.blend" --python-exit-code 1 -P hover_poses.py -- "$PWD" > hover-build.log
"$bl" -b small_chamber_sensor.blend --python-exit-code 1 -P assembly.py > assembly.log
"$py" -c 'import json,numpy as n;open("rig_origin.json","w").write(json.dumps(n.load("geometry.npz")["world_origin_m"].tolist()))'
"$py" walk.py insert insert
"$py" walk.py sand natural
"$py" acquire.py
"$py" punches.py
"$py" punches.py --merged
"$bl" -b small_chamber_sensor.blend --python-exit-code 1 -P backdrop.py -- "$src" "$PWD/room_insert" > room_insert.log
"$bl" -b "$src/marked_fly_small_chamber.blend" --python-exit-code 1 -P backdrop.py -- "$src" "$PWD/room_original" > room_original.log
for f in insert natural; do "$py" cameras.py "$f"; done
for f in cal blind punches merged hover insert natural; do "$py" estimate.py "$f" cal --rig-origin rig_origin.json; done
"$py" spatial.py
"$py" episodes.py insert
"$py" score.py
"$py" plot.py
"$py" video.py
"$bl" -b small_chamber_sensor.blend --python-exit-code 1 -P walking_scene.py -- "$src" "$PWD" > walking-scene.log
"$bl" -b walking_sensor.blend --python-exit-code 1 -P check_scene.py -- "$PWD" > reopen.log
"$bl" -b walking_sensor.blend --python-exit-code 1 -P render_checks.py -- "$PWD/fullrgb" > fullrgb.log
"$py" verify_images.py "$src" "$PWD"
