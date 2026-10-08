#!/bin/bash
set -u
SRC=/run/media/sensen/Data2/cell_wound_prototype
DST=/home/sensen/Desktop/cell_wound_prototype
mkdir -p "$DST/engine"
cp -v "$SRC"/engine/*.py "$SRC"/engine/*.md "$DST/engine/"
cp -v "$SRC/run_whole_fly.py" "$SRC/run_sensorimotor_verify.py" "$DST/"
cp -v "$SRC/data/flywire/fafb_sensorimotor_groups.npz" "$DST/flywire/" 2>/dev/null
for f in metrics_whole_fly.json whole_fly_record.json whole_fly_tracheal.png \
         metrics_sensorimotor.json; do
  [ -f "$SRC/outputs/$f" ] && cp -v "$SRC/outputs/$f" "$DST/"
done
echo; ls "$DST" | wc -l
