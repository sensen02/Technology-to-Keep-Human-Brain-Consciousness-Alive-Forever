#!/bin/bash
set -u
SRC=/run/media/sensen/Data2/cell_wound_prototype
DST=/home/sensen/Desktop/cell_wound_prototype
mkdir -p "$DST/flywire"
# code + cached connectome summary (NOT the raw 70MB csv, NOT any credential)
cp -v "$SRC/build_flywire_connectome.py" "$SRC/run_flywire_brain.py" "$SRC/flywire_download.py" "$DST/"
cp -v "$SRC"/engine/*.py "$SRC"/engine/*.md "$DST/engine/"
cp -v "$SRC/run_neural_selftest.py" "$SRC/run_engine_selftest.py" "$SRC/run_engine_demo.py" "$DST/"
for f in metrics_flywire_connectome.json metrics_flywire_brain.json \
         flywire_brain_record.json flywire_brain_activity.png \
         metrics_neural_selftest.json NEURAL_SELFTEST.zh-CN.md \
         ENGINE_SELFTEST.zh-CN.md metrics_engine_selftest.json; do
  [ -f "$SRC/outputs/$f" ] && cp -v "$SRC/outputs/$f" "$DST/"
done
cp -v "$SRC/data/flywire/fafb_connectome.npz" "$DST/flywire/" 2>/dev/null
echo; echo "交付完成"; ls "$DST" | wc -l; ls "$DST/flywire"
