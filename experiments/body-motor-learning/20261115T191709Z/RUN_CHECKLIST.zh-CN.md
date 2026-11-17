# 站姿回正与抗扰学习运行记录（S2/S5）

- 预登记 `PREREG.zh-CN.md`。固定上一轮物理实现与成对施力，旧站立网络、关闭学习分支和打开学习分支三臂，12 个新纯推力以种子 2026111531 整组留出，另做 3 项台架回归。输入、关节半范围、环境与资产身份存于 `RUN/IDENTITY.json` 和测试各进程的 `IDENTITY.json`。训练数据散列见 `RUN/DATA.json`。
- 归档解释器 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python`，启动参数 `-B -u`，环境 `MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`。
- 首启动工具会话 61340，退出 1，PyTorch 缓冲名冲突时尚未更新，保留 `20261115T191530Z/train.log`。修复后新工具会话 76512 依序运行 `bounded.py train`、`bounded.py eval --workers 8`、`bounded.py replay`，退出 0。工具会话 66581 运行 `summarize.py`，退出 0。
- 固定 800 次 Adam 模仿更新，所有权重、Adam/RNG 状态与对应日志均保存。行走、PPO 与正式颈部部署未执行。本实验接触力门为 1 ms 采样。未执行原生逐接触审计。
- 45 条测试动作回放、受力与目标脉冲配对、指标复验及 801 权重散列通过。n01 与 leg_pair 从完整快照的独立仿真保存值逐位相同。5,406,320 原生积分。留出图已目视，视频未生成。
- 12 组中 9 组各臂均安全恢复，有界学习分支加速其中 8 组。两组强推失支撑与一次非足触地保留为失败。完整结果、指标与产物见本目录报告。
