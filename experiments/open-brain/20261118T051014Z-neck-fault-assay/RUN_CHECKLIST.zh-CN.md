# 运行与复验

工作目录为 `/run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261118T051014Z-neck-fault-assay`。沿用上轮 `drive.sh`，实际 Python 为 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python`。Python 3.12.14、PyTorch 2.5.1 CPU、MuJoCo EGL，OMP、OpenBLAS、MKL 各一线程。

| 命令 | 用途或记录 |
|---|---|
| `bash drive.sh geometry_fault.py` | 退出 0。核验上一轮有限元输出哈希并重算注册轴突求交和截点 |
| `bash drive.sh derive.py` | 退出 0。复用固定颈部和双向 Eon 平地运行入口，向原颈段入队时加入逐轴突传导结果 |
| `bash drive.sh analysis/probe.py` | 退出 0。固定队列边界与间歇漏发均通过，见 `probe.log` 和 `QUEUE_PROBE.json` |
| `bash drive.sh run_fault.py 10901 FORMAL/seed10901` | 六臂闭环，退出 0，见 `FORMAL_10901.log` |
| `bash drive.sh run_fault.py 11311 FORMAL/seed11311` | 六臂闭环，退出 0，见 `FORMAL_11311.log` |
| `bash drive.sh run_fault.py 11701 FORMAL/seed11701` | 六臂闭环，退出 0，见 `FORMAL_11701.log` |
| `bash drive.sh run_fault.py 11701 REPLAY/seed11701` | 独立复跑六臂，退出 0，见 `REPLAY_11701.log` |
| `bash drive.sh analysis/audit.py` | 退出 0。从实际源端发放和独立计算的截点状态重建双 AN、17 DN 到达，验证对照与重放 |
| `bash drive.sh analysis/figures.py` | 退出 0。几何与脉冲图、平地配对轨迹图均已实际打开 |

每个结果目录是独立新目录。`INPUTS.json` 中的哈希固定源代码、连接组、原模型、权重和身体资产，每一个实际回合均留 `NEURAL.npz` 及各臂的 `TRACE.npz` 和验收记录。
