# S2/S5 真实物理滑移PPO：前缀历史与学习样本分离

目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T122634Z`，源角色GRU1000检查点在 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T120531Z/BC_GRU_ROLE_1000/final.pt`。该确定性9.6s保持六腿前进/足载/高度/非足0，但RH滑移.534655>.5，故启动预登记3update小步物理奖励优化，不修改验收阈值。

A. 完整历史960teacherprefix＋1200autonomous，重新从零hidden完整BPTT，不存旧hidden当训练初态。prefix学习mask0、autonomous1；actor/critic/adv归一化/KL/计数都只取autonomous，prefix真实teacher观测与动作历史仍影响hidden。teacher动作不冒充on-policy样本；后段从Gaussian采样，42输出只由ANN生成。

- BC→PPO仅载模型均值/gain，显式std初始化.02、bounds[-5,.3]，模型身份记录；BC Adam不带入PPO，actor/critic独立freshAdam。原critic权重保留但此阶段开始拟合真实return。
- PPO gamma exp(-.005)、GAElambda exp(-.005/.2)、clip.1、4epoch、actorlr3e-5/critic1e-4、maxnorm.5、42维jointKL.03，越界同时回退模型/两个Adam/计数并半LR。
- 奖励每1ms实际COMv/footload/contactspeed/nf/height算rate并积分，原生物理仍12.5us；不含相位/腿序或达到速度作分母。总rate下限−100显式记录raw各项与floor比例，unsafe早停后剩余最坏rate吸收惩罚仅回报标量，不伪造未来身体/动作。
- 74检查通过：实际masked PPO actor/gain/logstd/critic FD、全2160步replay/BPTT与前缀梯度、prefix排除/只autoGAE/KL/归一化、stale/NaN拒绝、两个非空Adam精确回退、截断后继bootstrap、floor/吸收、真实CycleTracker重置时机及mock teacher后段0调用。student81PASS与sourcepreflight独立通过，无长实际训练混入检查。
- 同完整71mesh派生身体/无附着/有限PD2rad/s/力矩/质量/重力/摩擦/12.5us物理与1ms积分，未写真实根或加力。原始完整验收与分支工程门区分。

B. root bash149正在运行3updates，每轮1s零稳站＋4.8s实际teacher＋6s纯ANN采样。第1update1200自主决策完整无终止、6腿均有事件，actor接受4步jointKL.01217，body三门/forward通过，采样slip仍失败；这不是最后确定性验收。逐更新保存原始episode/1mstrace/完整checkpoint与reward/adv/KL/counter，最后update选择不按分数覆盖已通过父teacher。

C/D/E. 实际bash149三update完成exit0，均1200ANN采样决策无终止、各4步更新接受、jointKL.01217/.00754/.00594，所有模型/双Adam/计数/源身份保存，原loader重载逐位一致；reward总floor比例.00117/0/.00117、rawslip分项未隐藏。bash151新进程final确定性9.6s纯ANN完成exit0，forward.0718777/COM1.255642/nf0/足载≈1/全6腿周期，但RHmaxslip.54405>.5且高于BC1000.53465，未提升为好模型。没有后段teacher查询/状态reset，失败保留，原始完整验收未完成。临时检查清理，goal继续。
