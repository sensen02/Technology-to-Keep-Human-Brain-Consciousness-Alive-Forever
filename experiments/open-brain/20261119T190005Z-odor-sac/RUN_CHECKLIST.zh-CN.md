# P1 嗅觉 SAC 复验清单

实验位于 `/home/sensen/.codex/worktrees/4528/cell_wound_prototype/experiments/open-brain/20261119T190005Z-odor-sac`。新目录用于受控研究，既有权重与场景保留原文件。训练环境 Python 3.12、Torch CPU，OMP、OpenBLAS、MKL 各 1 线程。物理环境使用现有 EGL 启动器、MuJoCo 3.1.6 和独立 CPU 神经进程。每次仅启动一对物理／神经进程。

```bash
run_dir='/home/sensen/.codex/worktrees/4528/cell_wound_prototype/experiments/open-brain/20261119T190005Z-odor-sac'
python_cpu='/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python'
body_runner='/home/sensen/.codex/worktrees/4528/cell_wound_prototype/experiments/open-brain/20261119T012530Z-survival-orchard/run_env.sh'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$python_cpu" -uB "$run_dir/reachability.py"
"$python_cpu" -uB "$run_dir/train.py" "$run_dir/REPLAY_TRAIN" --steps 110000 --interval 10000
"$python_cpu" -uB "$run_dir/train_assisted.py" "$run_dir/REPLAY_PROXIMITY" --steps 170000 --interval 10000
"$python_cpu" -uB "$run_dir/train_assisted.py" "$run_dir/REPLAY_DETOUR" --steps 140000 --interval 10000 --course detour --guided
"$body_runner" "$run_dir/live.py" "$run_dir/REPLAY_OPEN_ODOR" --policy "$run_dir/TRAIN/POLICY.npz" --stage open --sense odor --duration 6.4
"$body_runner" "$run_dir/live.py" "$run_dir/REPLAY_OPEN_BLANK" --policy "$run_dir/TRAIN/POLICY.npz" --stage open --sense blank --duration 3.2
"$body_runner" "$run_dir/live.py" "$run_dir/REPLAY_DETOUR_L" --policy "$run_dir/TRAIN_DETOUR/POLICY.npz" --stage detour --side 1 --seed 89217 --sense odor --duration 6.4 --assisted
"$body_runner" "$run_dir/live.py" "$run_dir/REPLAY_DETOUR_BLANK" --policy "$run_dir/TRAIN_DETOUR/POLICY.npz" --stage detour --side 1 --seed 89217 --sense blank --duration 3.2 --assisted
"$body_runner" "$run_dir/live.py" "$run_dir/REPLAY_DETOUR_R" --policy "$run_dir/TRAIN_DETOUR/POLICY.npz" --stage detour --side -1 --seed 89217 --sense odor --duration 6.4 --assisted
"$python_cpu" -uB "$run_dir/audit.py"
```

第一次单独探测以未挂载 FlyBody 源路径的方式导入退出 1，改为使用归档入口后识别真实的左右触角刚体。后移墙的首次训练因几何循环变量覆盖评估的场地名称，在第一个存档日志处退出 1。失败保存在 `TRAIN_DETOUR_INITIAL`，修复后从相同随机种子重新训练并重新选择模型。

三条正式训练、可达性测试、五次物理运行、独立复载、图生成、源码解析、哈希和 `git diff --check` 均退出 0。五次物理运行顺序为平地气味、平地关气味、左绕气味、左绕关气味、右绕气味。物理记录累计 432000 原生步与 216000 神经拍。实际打开了物理轨迹图和左绕时间序列。 `RELOAD.json` 记录选权重和 48 留出完整轨迹逐值复现，`PAIRING.json` 记录两组消融首个感觉采样前的匹配。
