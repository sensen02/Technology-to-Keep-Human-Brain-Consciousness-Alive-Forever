# 感知记忆／融合闭环失败诊断：仅代码准备，不训练

实际标准库33项检查通过exit0，仅读6份实际记录的ZIP/NPY头核字段和维数，没有数组payload、NumPy/Torch、模型、Core、渲染或物理执行。原所有科学源不改。

父CPU1入口：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T045302Z/failure_diagnostic/diagnose.py --execute-parent > /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T045302Z/failure_diagnostic/RUN.log 2>&1
```

固定权重db2510d46900644c7f8c7b556b4b892dad2a88e3e37ce13792811916a928d4c6及原冻结66f，不更新任何参数。完整6案例：O_left／O_left_odorcut／O_noodor／F_left_agree／F_right_agree／V_empty。

先逐步独立66f重放实际actor_receptors，对odor隐藏状态前／后、raw、latent逐位核验。报告窗口pre .82—1.6、earlycut 2.1—2.6、late3—3.5、last8.6—9.6秒的浓度／环境受体／actor受体／嗅觉和视觉神经raw／解码turn／AN-DN接收及脑和嗅觉记忆范数。Ocut受体归零而感知记忆保持，是与直接归零odor神经标量不同的干预。

四种感官输入历史探针：冷零受体、实际Ocut、实际Oleft驱动史后1.6秒强制零受体、实际Fleft驱动史后零受体。均保持嗅觉记忆连续，不运行时清h或turn。若零受体后raw仍大，支持感官网络持久化这个具体观察，不凭此直接认定双稳态或唯一根因。

另对同一实际Ocut保存身体观察作四次源消息队列重放：baseline、仅1.6秒后脑的odor神经标量为零、仅AN接收基线零／拒绝该方向消息、两者同时。保留源队列实际1ms推进15400tick和暖段AN历史，h16全程连续。baseline的脑状态、AN-DN编码、接收活动及解码turn必须逐位复现实际记录，切断前1480条前缀精确；消息2ms释放、5ms消费。此重放不重新计算身体／motor轨迹，属于固定观察条件的通信敏感性，不是物理反事实。

训练域比较使用实际已保存27序列的body111／双模态raw／教师范围、最终存储脑状态与AN，和实际6案例的归一化范围／脑norm／AN对比。仅报告实际值，不设新通过阈值，不训练新模型，不把离线教师强度公式放入运行控制。F_left_agree循环缺LF/LH/RH的事实从原RESULT读取，支持／COM／非足通过而周期失败不等同倒地。

所有输出exclusive，失败时保留已写INPUTS／OUTPUTS／RESULT；先读日志，不盲重跑。源和参数SHA前后核验，0optimizer／0Core／0新增物理。后续修复设计需父根据实际诊断另作决定，本目录不包含候选、训练或恢复运行规则。
