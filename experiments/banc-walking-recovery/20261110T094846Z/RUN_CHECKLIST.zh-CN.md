# S2/S5 运行检查记录：实际偏离状态辅助采集与恢复模仿

A. 本轮假设：只沿教师轨迹拟合缺少闭环误差状态。父模型 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T094051Z/BC_FF_SMOOTH_1000/final.pt`，新目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T094846Z`。PREREG.json先行。身体/原生时间/执行器不改；actor原238输入不加相位或教师内部状态，不使用标准化以独立检验恢复数据。模型/源码/环境/种子/编译物理身份在各IDENTITY，旧新源码在CHANGE_LEDGER。

- 已验证教师参考 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T093347Z/early_unload`，同样配置在两个小扰动回合通过。
- RECOVERY_CHECKS全过：实际动作/上一动作关联、连续教师与隐藏历史、5ms保持、早停、train/validation隔离、完整GRU前缀梯度、权重与fresh optimizer/RNG重载、输入篡改拒绝；无长真实训练混入检查。student评估74检查实际20ms保持/撤辅契约另已通过。
- 运行入口本目录env/bin/python -B，OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1。采集episodes3，seed20261201..3，最后一个验证；每次1s零稳站＋20ms小电机脉冲，4.8s纯教师＋4.8s .9expert/.1student＋.01噪声。只42受限变化率驱动，不写根/外力。
- 标签是在实际每个偏离状态查询的教师动作，原X含上一实际混合动作。保存完整前缀与混合序列；phase仅来源字段。辅助混合不计自主。COM<1.2或非足>0仅采集安全早停，不改变行走验收。

B. bash-86采集退出0，3回合各1920决策、mixed960，无早停。报告stdout过大问题已记录，下一版本缩日志，不修改已运行源。bash-88模型权重warmstart、fresh BC Adam/RNG、1000步lr.0003、base+recovery样本等权，退出0。原/恢复验证回合都不训练。日志每25步，CURVE、final.pt及身份关联保存，保存后重载一致。

C. 独立验证 RECOVERY_PHYSICAL_CHECKS.json核验metadata/hash/trace：混合阶段COM最低≥1.2558/非足0/前向约.0716/滑移≤.4903，但其控制源.9teacher因此不能叫神经通过。第一次评估bash-90在仿真前因缺STUDENT_CHECKS退出1；运行未改检查74PASS后同命令bash-91退出0。1s站稳＋4.8s教师＋9.6s纯student，连续状态/target/previous/hidden，无teacher查询；前向-.07422、足载.12674、COM.67754、非足46.689原生力、滑移2.425，失败。失败轨迹/命令/实际动作均保存，没有挑短窗忽略跌倒。

D. 最终成功视频未执行：神经未通过。ROUND2_ACTUAL_STATES.png为同次实际保存状态，71mesh全身不透明，已打开核验；不另造动作。旧算法教师视频保持明确标签，不能冒充本次自主。

E. 所有任务已收集，无残留活跃仿真/训练。只读诊断一次NPZ循环解压超时、后继取消；数组缓存后退出0，不影响输入。临时检查目录自动清理，失败和必要权重/数据保留。阶段报告 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T094846Z/ROUND2_REPORT.zh-CN.md`，下一最小动作增加采集学生参与比例或短学生爆发段，扩大实际误差状态覆盖，继续独立纯student验收。
