# 站立示范扩充运行记录（S2/S5）

- 身份与配置存于 `RUN/{dev,train,on,test}/CONFIG.json`、各工作进程的 `IDENTITY.json` 和 `PREREG.zh-CN.md`。沿用 71 mesh 独立站立身体、无附着、42 路有限增量 PD，作者控制只在采集窗。
- 归档解释器 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python`，启动 `-B -u`，环境 `MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`。
- 顺序 `collect.py dev`、`collect.py train --workers 12`、`learn.py initial`、`collect.py on --workers 6`、`learn.py finish`、`learn.py test --workers 8`、`learn.py replay`、`audit.py`，均退出 0。工具会话 90782、31231、45536 已执行。
- 数据先按全轨迹分组，教师轨迹逐条验收，来源及 SHA 记录于 `RUN/learn/DATA_1000.json`。接续分支首段匹配续跑分支，现场力及力矩、初态和扰动速率跨臂相同。
- BC 优化两段 600+400 次，固定保存全部 Adam/RNG 与 SHA。PPO、GAE 和正式走路不适用。最后检查点不采用事后最佳步。20,804,640 原生积分。接触指标以 1 ms 采样，原生逐接触审计未执行。
- 独立测试动作及全轨迹复验通过，行为失败。完整失败列表和诊断见本目录报告。留出图已目视，视频未生成。
