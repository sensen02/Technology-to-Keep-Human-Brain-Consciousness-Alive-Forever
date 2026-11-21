# 本轮复验清单

运行目录为 `/run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261118T044947Z-neck-insertion-assay`。以下命令从该目录运行。`drive.sh` 使用 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python`。Python 3.12.14，PyTorch 2.5.1 CPU，MuJoCo EGL，OMP、OpenBLAS 和 MKL 各一线程。新增求解器只装进本实验的 `solver_deps/`，安装退出 0。

| 命令 | 记录 |
|---|---|
| `bash drive.sh derive.py` | 生成并核对从既有运输层派生的接触时序接口 |
| `bash drive.sh mechanics.py` | 退出 0。四组求解，记录在 `mechanics.log`；保存形变、弧长、接触间隙、方程残差 |
| `bash drive.sh cable.py` | 初次工程电缆激发未获传播事件，退出 1，见 `cable_first_failure.log`。确认两端各一次传播越阈后重跑退出 0，见 `cable.log` |
| `bash drive.sh sensitivity.py` | 退出 0。比较轴突直径假设、电缆步长、力学网格和接触条件 |
| `bash drive.sh half_contact_check.py` | 退出 0。比较名义位移和一半位移 |
| `bash drive.sh plot_results.py physical` | 退出 0。输出 `MECHANICS_AND_DELAY.png` |
| `bash drive.sh run_live.py 10901 FORMAL/seed10901` | 四臂闭环，退出 0 |
| `bash drive.sh run_live.py 11701 FORMAL/seed11701` | 四臂闭环，退出 0 |
| `bash drive.sh run_live.py 11701 REPLAY/seed11701` | 独立进程重放，退出 0 |
| `bash drive.sh run_live.py 11311 FORMAL/seed11311` | 前两次收到 SIGTERM，退出 143，分别只到已打印的 10,000 与 15,000 拍。中止证据在 `INTERRUPTED/`。用相同参数独立进程重跑，退出 0，见 `live11311.exit` |
| `bash drive.sh audit_live.py` | 退出 0。双 AN、17 DN 全量重构到达事件，对照与归档相等，53 个重放数组相等，见 `audit.log` 和 `FINAL_ASSAY.json` |
| `bash drive.sh plot_results.py behavior` | 退出 0。输出 `MATCHED_FLAT.png`，与 `MECHANICS_AND_DELAY.png` 均已实际打开 |

平地每臂运行 20,000 步。输入收据固定权重、BANC/FlyWire 身份、身体资产、控制器、模型参数与刺激。单位、最大应变、身体安全和比较门槛见同目录 `PROTOCOL.zh-CN.md` 和 `REPORT.zh-CN.md`。
