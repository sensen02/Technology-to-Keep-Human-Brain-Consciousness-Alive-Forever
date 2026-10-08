#!/bin/bash
set -u
SRC=/run/media/sensen/Data2/cell_wound_prototype
DST=/home/sensen/Desktop/cell_wound_prototype
mkdir -p "$DST/engine"
cp -v "$SRC"/engine/*.py "$SRC"/engine/*.md "$DST/engine/"
cp -v "$SRC/run_cable_selftest.py" "$DST/"
for f in metrics_cable_selftest.json; do
  [ -f "$SRC/outputs/$f" ] && cp -v "$SRC/outputs/$f" "$DST/"
done
echo; echo "交付文件数: $(ls $DST | wc -l)"
