# S2/S5 六足速输入神经连接的显式剪枝对照

目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T123957Z`，父1000GRU检查点滑移.534655略高门槛。此前300模型撤辅后断开6足速通道使滑移1.351→1.193但未通过；本轮是独立神经权重干预，不继承那个结果，也不修改.5门槛。

A. 导出只将actor.encoder.weight[:,189:195]的768条输入突触权重置0，所有其他actor/critic/std/模型buffer逐位不变，原输入238维/原有限PD42输出与物理不改。静态剪枝在预热和自主都存在，区别于旧回合只在撤辅边界断开；副作用由新实际交接测量。

- 所有原始感觉仍记录，原raw238输入契约和模型类不变；权重矩阵本身表达断开，没有推理外置滤波、脚本周期或输出控制器。剪枝不是梯度训练，元数据明确不能冒称原训练权重。
- EXPORT_CHECKS证明1920步原网络输入6通道全0与新网络读raw输入的mean/最终hidden逐位相同，模型state仅encoder.weight变化，6输入Jac0/其他物理输入梯度非零，完整新检查点原loader重载一致、无物理step。parent/new encoder hash、parent全checkpoint hash与source/compiled身份记录。
- 不声称旧BC Adam可精确续训，PPO状态空且独立；原BC训练依据在不可变parent保留。student81检查通过后实际eval。

B/C. root bash150实际同次1s站稳/4.8steacher priming/9.6s纯剪枝ANN完成exit0并收集，后段无teacher/head查询；forward.0718354/COM1.256556/nf0/足载≈1/六腿周期LF1其余2，但滑移.539728>.5且比未剪枝1000.53465差，未改善，不晋升候选。原始输入/actualsource/hidden/身体记录与身份完整，完整71mesh/无附着/质量重力摩擦/有限PD2rad/s/力矩/12.5us1ms不改。

D/E. 若未过全部门即失败候选，仍保留。成功候选需要独立重放/原生验收/同次连续视频，不能跳过。临时检查fixture无额外残留，目标继续。
