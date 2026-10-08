#!/bin/bash
set -u
SRC=/run/media/sensen/Data2/cell_wound_prototype
DST=/home/sensen/Desktop/cell_wound_prototype
mkdir -p "$DST/engine" "$DST/flywire"
cp -v "$SRC"/engine/*.py "$SRC"/engine/*.md "$DST/engine/"
cp -v "$SRC/build_banc_cns.py" "$SRC/run_virtual_env.py" "$SRC/run_receptor_selftest.py" "$DST/"
cp -v "$SRC/data/flywire/banc_connectome.npz" "$DST/flywire/" 2>/dev/null
for f in metrics_banc_cns.json metrics_virtual_env.json virtual_env_record.json \
         virtual_env_loop.png metrics_receptor_selftest.json; do
  [ -f "$SRC/outputs/$f" ] && cp -v "$SRC/outputs/$f" "$DST/"
done
echo; echo "交付文件数: $(ls $DST | wc -l)"; du -sh "$DST"
