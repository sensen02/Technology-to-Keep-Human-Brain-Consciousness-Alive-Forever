# 多初态站立与接触事件运行记录（S2/S5）

- 目标及配对口径见 `PREREG.zh-CN.md`。身体为冻结平地三球版本及 42 路有限增量 PD，附着关闭。权重为 b53bafa8c9b843c00c1162b1c55f46faf80354afb970ae1d13f6551c764793d3。
- 输入身份见 `RUN/INITIAL/IDENTITY.json`、`RUN/MATRIX/CONFIG.json`、各工作进程的 `IDENTITY.json`。每个初态、准备轨迹及测试轨迹留独立哈希。无优化器更新。
- 使用归档解释器 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python`，参数 `-B -u`。环境 `MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`。
- `snapshots.py`、`matrix.py run --workers 10`、`matrix.py replay`、`report_matrix.py` 退出 0。会话分别为 67398、36783、24724。共 7,544,000 原生积分。两个旧条件的两臂重叠数组逐位匹配。两条新进程物理复跑、32 对配对与 64 条数值记录重放通过。
- 每个原生步保存六足合力，所有非足有载接触保留身份、位置和世界力。矩阵和原接触数值保留。补记倒地定义、时间及探索性边界见 `EVENT_AMENDMENT.zh-CN.md`。
- 已打开两张实测图检查，视频未生成。行走、PPO 和颈部部署不在本轮站姿台架范围。
- 翻正快照导出前两次分别为导入顺序、结果计数键错误，退出 1，日志和部分产物保留。会话 74894 执行 `export_righting_seeds.py` 退出 0，280,800 原生积分，立项快照的完整恢复逐位确认。成功阶段累计 7,824,800 原生积分。失败额外积分 180,400 单列。
