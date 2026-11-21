# P1 新版果园迁移复验清单

工作目录 `/home/sensen/.codex/worktrees/4528/cell_wound_prototype/experiments/open-brain/20261119T194948Z-orchard-odor-transfer`。场景 `/home/sensen/orchard-3d/simulation/orchard.blend`。权重 `/home/sensen/.codex/worktrees/4528/cell_wound_prototype/experiments/open-brain/20261119T190005Z-odor-sac/TRAIN_DETOUR/POLICY.npz`。Blender 4.5.16 LTS CPU 双线程。物理为 Python 3.12、MuJoCo 3.1.6、EGL，神经 CPU 单线程。每次只运行一对物理／神经进程。

```bash
run_dir='/home/sensen/.codex/worktrees/4528/cell_wound_prototype/experiments/open-brain/20261119T194948Z-orchard-odor-transfer'
blender_run=/run/media/sensen/_dde_data/orchard-3d-work/tools/blender-4.5.16-linux-x64/blender
body_runner='/home/sensen/.codex/worktrees/4528/cell_wound_prototype/experiments/open-brain/20261119T012530Z-survival-orchard/run_env.sh'
mkdir "$run_dir/REEXPORT"
env OPENBLAS_NUM_THREADS=1 LIBGL_ALWAYS_SOFTWARE=1 \
  "$blender_run" --disable-autoexec --background /home/sensen/orchard-3d/simulation/orchard.blend \
  --threads 2 --python-exit-code 1 --python "$run_dir/export_crop.py" -- "$run_dir/REEXPORT"
"$body_runner" "$run_dir/preflight.py"
bash "$run_dir/run_suite.sh" "$run_dir/REPLAY"
"$body_runner" "$run_dir/preflight.py" heightfield
"$body_runner" "$run_dir/transfer.py" CLEAR "$run_dir/REPLAY/LIVE_CLEAR_HEIGHTFIELD_ON" --duration 4.8 --support heightfield
```

先期失败证据保留。首轮射线检查混入飞蝇自身几何，复验分配独立几何组。早期高度采样图碰撞实现出现约 .01 mm 像素重采误差，正式两组从源土三角面裁剪。随后查到试验脚本复用可变动作数组，修正并保持正式闭环和预检的逐步复制方式。初代重复使用动作的站姿结果仅用于诊断。

正式导出、网格与高度格预检、四个配对回合、一个高度格回合、独立审计、源码解析、图像、哈希和 git diff --check 最终退出 0。初次配对汇总将 NumPy 整数写入 JSON 退出 1，类型更正后在原轨迹上复算。五条物理记录、导出几何和源场景比对已完成，实际打开了轨迹图。
