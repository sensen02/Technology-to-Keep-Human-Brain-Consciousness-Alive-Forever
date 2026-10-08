#!/bin/bash
set -u
SRC=/run/media/sensen/Data2/cell_wound_prototype
DST=/home/sensen/Desktop/cell_wound_prototype
mkdir -p "$DST/engine"
cp -v "$SRC"/engine/*.py "$SRC"/engine/*.md "$DST/engine/"
cp -v "$SRC/run_engine_selftest.py" "$SRC/run_engine_demo.py" "$DST/"
cp -v "$SRC/run_neural_selftest.py" "$DST/" 2>/dev/null
for f in metrics_engine_selftest.json ENGINE_SELFTEST.zh-CN.md \
         engine_demo_record.json engine_demo_report.zh-CN.md \
         engine_selftest_record.json metrics_neural_selftest.json \
         NEURAL_SELFTEST.zh-CN.md; do
  [ -f "$SRC/outputs/$f" ] && cp -v "$SRC/outputs/$f" "$DST/"
done
echo; echo "engine/ 内容："; ls "$DST/engine"
