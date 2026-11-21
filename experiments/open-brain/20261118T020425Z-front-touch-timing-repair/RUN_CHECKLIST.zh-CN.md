# 前足上行时序修复实验记录

- 目标为P1、S1/S3/S5。父运行分别为 `/run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261118T002600Z-ascending-behavior-comparison/` 和 `/run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261118T014024Z-sensory-timing-check/`。本轮判据 PROTOCOL.zh-CN.md，保留旧报告、固定参数和所有历史轨迹。
- 闭环从同一编译身体及站姿开始，各自身体反馈。Python 3.12.14、Torch 2.5.1 CPU、OMP/OPENBLAS/MKL 各1线程、MuJoCo EGL。身体模型、原脑、颈固定参数、固定轴突和受体靶点身份见各运行 INPUTS.json/RESULT.json。同一源 AN 模型明确由 `timing.py` 选择，主修复保留已登记下行反馈。
- 从本轮目录通过 `bash drive.sh derive_live.py` 生成运行脚本，`bash drive.sh diagnose.py` 核对四位同负载样本、切断和固定回馈电导。相同入口 `bash drive.sh live_repair.py 10901 FORMAL/seed10901`、`bash drive.sh live_repair.py 11311 FORMAL/seed11311`、`bash drive.sh live_repair.py 10901 REPLAY/seed10901`。输出日志分别为 live10901.log、live11311.log、replay10901.log。接受和图用 `bash drive.sh accept.py`、`bash drive.sh plot.py`。最终六类步骤均 exit 0。
- 初次派生漏删遗留常量导致断言退出，未启动仿真。初始假定两根AN没有跨颈DN突触，正确性断言发现真实两边，在加载阶段退出，两次初始日志和空运行目录存入 setup_failure。改成保留这两条边的实际突触电导后开始正式步骤。历史失败不覆盖。
- 两次正式各4身体，独立新进程另4身体。共240000原生身体步、60000脑批tick/240000脑lane tick，同量实时颈批tick/lane tick。另有两次20000步切断对照和250批步双路电导验证，共40500诊断颈lane tick。四份前足率对照重用已锁定的源记录。无教师关节注入或PPO/GAE/训练。
- 正式覆盖原双相位。临界终点偏差≤.5mm、偏航≤max(10°,25%)及原速度、高度、安全门，另核推进和站稳。主修复受控参考2/2，完整原脑工程参考1/2。轨迹与神经新进程共52数组逐值相同。新生成图 ARRIVAL_TIMING.png 和 CLOSED_LOOP.png 已实际打开。此任务为感觉时序与既有动作回归，未验完整D1–D8、长时任务、全地形及实体生理；未制作视频。
- 按跑分和证据记录校验源哈希，旧入口失败状态保留。剩余问题是完整原脑工程参考的接口差，下一隔离试验是同一批颈后事件在原脑的入脑格式和原有14条AN回馈边界的影响。
