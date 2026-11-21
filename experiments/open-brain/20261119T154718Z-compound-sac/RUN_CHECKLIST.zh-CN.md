# P1 S4、S5 复合地形训练复验

- 工作目录为 `/home/sensen/.codex/worktrees/4528/cell_wound_prototype/experiments/open-brain/20261119T154718Z-compound-sac`。CPU 线程限制为 1。Python 3.12.14、Torch 2.5.1+cpu。身体与双眼使用同一父实验的 MuJoCo 3.1.6 和 EGL 启动器。
- `train_timed.py BUDGET_1800 --seconds 1800 --interval 120` 退出 0。输入及源码见 `TRAIN_INPUT_SHA256.txt`。每次保存的策略和优化器状态、开发集及摘要见 `CHECKPOINTS.jsonl`。独立重放结果见 `RELOAD_AUDIT.json`。
- `check_geometry.py`、`inspect_vision.py`、`physical_suite.py LIVE_BASELINE`、两次 `visible_food_probe.py` 和 `bash run_selected.sh` 最终退出 0。冻结权重、左右镜像、同初始站姿、食物库存和运动读出均有输出。候选共 5 个原场景和 2 个可见轮廓回合。与基线累计 14 个实时回合。
- 初次脚本修正保留日志。碰撞审计最初读取库存对象而非资源表。基线审计最初将 0.1 ms 的神经采样间隔当成了 0.05 ms 的物理步长，复核原数据后修正。静态视图初次缺少路径初始化。危险离线参照初次读了整个关节向量，已切片根四元数。浮点导出初次压力门失败，独立复核确认为 float32 舍入；实景观测保留原 5 × 10⁻⁶ 门，同时逐值检查权重和 float64 对照。
- 已实际打开场景、起点双眼、训练图、实景宽桥身体回放、可见轮廓身体回放、轨迹和风险时序图。相关源码通过 `py_compile`，脚本通过 `bash -n`，工作树通过 `git diff --check`。
- 限定闭环宽桥达到摄入 0.18 和末 140 ms 站稳。四个复杂原场地完整觅食为零。额外可见轮廓试验 0/4 通过第一道沟。下一步定位单桥斜坡的连贯风险和沟前余量，复用本轮站姿、种子和警戒线。
