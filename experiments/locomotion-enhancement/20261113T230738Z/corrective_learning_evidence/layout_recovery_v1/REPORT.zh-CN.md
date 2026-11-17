# 三候选训练拟合与独立身体验收

该图只描述实际 final600 在保存训练输入上的动作拟合，不代表闭环转向、站立或行走验收。三个候选的共享八源使用相同训练目标；第三候选这些序列全部1920标签有效。

- stop：C1 residual96=0.06058844；C2 fastpath=0.04366385；C3 local mean=0.02835800
- left_pivot：C1 residual96=0.11003220；C2 fastpath=0.08849112；C3 local mean=0.06706112
- right_pivot：C1 residual96=0.10163915；C2 fastpath=0.08077543；C3 local mean=0.06161978
- walk：C1 residual96=0.01742189；C2 fastpath=0.01444868；C3 local mean=0.01489491
- V_left_pulse：C1 residual96=0.01736359；C2 fastpath=0.01602458；C3 local mean=0.01760763
- V_right_pulse：C1 residual96=0.01791851；C2 fastpath=0.01657686；C3 local mean=0.01780240
- V_left_to_right：C1 residual96=0.02070698；C2 fastpath=0.01911685；C3 local mean=0.02086320
- V_right_to_left：C1 residual96=0.02069055；C2 fastpath=0.01913555；C3 local mean=0.02002601

额外四源单列，不混入共享八源或泛化成绩。辅助源使用未混合教师目标，但采集过程中有教师辅助，不能据此称纯ANN身体通过。

- old_walk_control：合格标签 1920/1920，masked RMSE=0.01510983；旧控制直接目标
- assisted_stop：合格标签 1920/1920，masked RMSE=0.03052563；教师辅助目标
- assisted_left_pivot：合格标签 1280/1920，masked RMSE=0.07002565；教师辅助目标
- assisted_right_pivot：合格标签 1293/1920，masked RMSE=0.06205718；教师辅助目标

纯身体验收使用原始独立RESULT/VERIFY及父审核记录，图中逐任务保留FAIL/PASS与实际失败门。第三候选 已按显式提供审核路径核验，详见JSON。

所有来源SHA前后一致。未读取NPZ载荷、未加载模型、未执行NN forward/Core/物理；仅父执行Matplotlib绘图。

图像：/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T230738Z/corrective_learning_evidence/layout_recovery_v1/TRAINING_FIT_COMPARISON.png
机器审核：/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T230738Z/corrective_learning_evidence/layout_recovery_v1/PLOT_VERIFY.json
父实际目视审核：PENDING。
