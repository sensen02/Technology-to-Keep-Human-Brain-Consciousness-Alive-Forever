# S2/S5 续轮3：50%学生参与的实际状态聚合

## A. 启动前

新目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T102045Z`，父版本 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T094846Z`。PREREG.json固定目标、预算与失败条件：90%辅助数据偏离示范约.055每物理通道标准化RMS，而撤辅学生偏离更大；本轮只提高采集学生比重，不同时改变actor输入/教师/物理。

- 控制源明确为训练用教师/学生混合，不能计自主。完整71mesh身体、无附着、原有限PD/2rad/s/力矩与1ms目标积分/12.5微秒物理不变。
- parent FF模型只读加载：`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T094846Z/BC_FF_RECOVERY/final.pt`。
- 计划实际采集3独立seed20261221..23、最后一个验证，1s零稳站/20ms小电机脉冲/4.8s教师前缀/4.8s beta=.5混合与.01噪声；COM<1.2或非足>0逐1ms采集早停、失败前缀保留。phase仅标签来源，不给actor；X上一实际动作、Y教师当前实际状态动作。
- 源/旧新hash在CHANGE_LEDGER与运行IDENTITY；模型保存关联全部基线/恢复数据、父权重身份。
- 适用检查：beta/动作混合与元数据一致、5ms保持、无reset/实际前动作连续、GRU完整历史/前缀梯度、独立验证、freshBCAdam/RNG/PPO、状态重载与篡改拒绝。检查完成前不运行真实采集。未做新PPO学习，本轮不需要重改GAE或损失。
- 所有命令使用本目录env/bin/python -B，OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1。训练若执行按MODELONLY起点、base+recovery样本等权1000步lr.0003、seed20261231，之后同一连续接管协议纯student9.6秒独立评估。

## B—E. 执行与结论

实际任务bash92采集三条1920决策完成exit0、无早停；混合滑移.508–.515超门，仅训练数据。RECOVERY_ANALYSIS核验全部hash/时序/prevaction、覆盖RMS.065–.068。预登记AMENDMENT_BUDGET_CONTROL后bash93旧数据等预算对照、bash94新数据训练各1000步lr.0003seed20261231完成exit0/reloadexact。bash95/96新进程真实连续接管都失败：COM最低.50727/.50496，非足49.77/40.488原生力，足载下降；所有状态交接连续。all4IDENTITY独立verify通过。ROUND3_COMPARISON.png已打开，失败保留。全目标原始验收和最终成功交接视频仍未完成；物理安全与门槛不降低。
