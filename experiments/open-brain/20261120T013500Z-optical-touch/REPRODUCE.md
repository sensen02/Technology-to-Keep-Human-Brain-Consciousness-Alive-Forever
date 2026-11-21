# 复验

在本实验目录打开 shell。只复算原动画 66 行不需要 Blender，输出文件须是新路径。

```bash
src=/home/sensen/Desktop/engine修复验收/八机位果蝇观测实验室-小箱与关节标记
py=/run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261117T051249Z-bidirectional/flight_env/bin/python
bl=/run/media/sensen/_dde_data/orchard-3d-work/tools/blender-4.5.16-linux-x64/blender
out=$(mktemp -d)
export OPENBLAS_NUM_THREADS=1
"$py" infer.py "$src/input.json" "$src/reconstructed.npz" empty "$out/predictions.csv"
cmp predictions.csv "$out/predictions.csv"
"$py" check_missing.py "$src" empty
```

重新拍空箱、拍量规和执行事后几何对照时使用独立目录。几何对照只接受已经有预测及哈希的目录。

```bash
"$bl" -b "$src/marked_fly_small_chamber.blend" --python-exit-code 1 -P capture.py -- "$src" "$out/empty"
"$py" infer.py "$src/input.json" "$src/reconstructed.npz" "$out/empty" "$out/fresh.csv"
cp "$out/fresh.csv" "$out/predictions.csv"
(cd "$out" && sha256sum predictions.csv > predictions.sha256)
"$bl" -b "$src/marked_fly_small_chamber.blend" --python-exit-code 1 -P compare_geometry.py -- "$src" "$out"
"$bl" -b "$src/marked_fly_small_chamber.blend" --python-exit-code 1 -P fixture.py -- "$src" "$out/gauge"
"$py" gauge_track.py "$src" "$out/gauge/marked" "$out/gauge/tracks.npz"
"$py" infer.py "$src/input.json" "$out/gauge/tracks.npz" "$out/gauge/empty" "$out/gauge/predictions.csv"
```

assess.py 在脚本目录读取已归档的原试验、量规结果、几何对照和独立复算的哈希，写出统计、图和状态变化表。
