# 本轮实施检查记录

目标：S2/S5，按已定规范实施行走改进。实验族绝对路径：`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T045410Z`。

## 身份与权限

- 本轮已获用户“实施吧”授权；无真实硬件/外部发布。goal已启用，id为goal-36285f1d-056f-40cb-a2da-060692fcfcf2。
- 身体沿用已声明的派生三垫/x200模型，摩擦1.5、每加载腿附着下拉0.25体重，增量目标±2rad/s；不是冻结身体或原连接组能力。
- 原始源码只读引用、关键控制源码已复制到本目录sources；运行实际依赖、XML/资产/初态/模型/编译后数组在每个子运行IDENTITY.json保存SHA-256。
- 环境env是独立venv，原物理依赖通过只读引用的.pth复用。torch2.5.1+cpu写入新env，原venv_body未修改。NumPy2.5.3、MuJoCo3.9.0、FlyGym2.1.0，NumPy↔torch转换检查退出0。

## 初次实验设计

- 阶段0：真实损失/GAE/Adam回滚/完整checkpoint检查、旧新物理一致性、纯观测、原生记录读回、旧权重独立重放。
- 修正前馈短基线：ff495，初始化为旧保存best第9次（不是遗失的BC权重），保持其探索方差；1ms策略、2秒有限时域、每update至少8秒完整回合，3updates。
- 该基线保留旧当前源码的奖励代理；自动微分、分开的梯度裁剪/Adam、回滚与保存机制同时修正，不能归因于单一bug修复。
- 只有两份源码hash匹配的检查报告明确通过才允许训练。每更新保存checkpoint和对应评估；未原始全验收，不发布行走通过结论。
- 数值/物理警告、写盘失败或运行输入hash变化将中止依赖实验并记录原因；不带错继续长训练。

## 当前证据路径

- `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T045410Z/PREREG.json`
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T045410Z/LEARNING_CHECKS.json`
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T045410Z/PHYSICS_CHECKS.json`

报告和完成状态以实际JSON及退出码为准；本记录不预先勾选未执行项，不把已实现的GRU推理API当成已经完成循环PPO训练。后续需在独立子运行固定配置后逐步对照。
