# 感觉编码器输入边界修复

手工研究原型，不是成品；此修复不宣称改善站立。

## 可重现实现缺陷

构造感觉组时直接astype整数会将浮点神经索引静默截断；布尔索引也会变成0/1。错误维度/重复索引缺少显式检查。归一化分母为零时出现临时ZeroDivisionError，NaN/Inf的配置与物理量可传播进网络。新增回归测试先暴露了这些问题，再修复输入边界。

## 兼容性要求

保留有限非负增益（包括0）、非负整数神经元数（包括空集合0）、有符号物理量、不裁剪接触输入、跨通道/跨腿显式共享索引。只拒绝组内重复、非整数/布尔索引、非一维数组、非有限量和无效分母。不改变默认编码公式、极性、正常电流或网络参数。

## 未纳入本次的历史API问题

assign(leg,...)只有单个_rows而不是按腿保存，legacy参考姿态要求>=66。当前真实闭环通过canonical映射直接生成ChannelQuantities后调用encode，不走assign/quantities旧接口，因此这些遗留问题不是当前下落的已证实原因，暂不混入此修复。

## 验证

修复前测试暴露49个失败子项和3个零分母错误；修复后17项输入边界测试与4项选择性电流测试全部通过。编码器每次encode重新校验可变配置，并拒绝归一化、乘法或共享累加导致的非有限结果，不裁剪掩盖异常。

7组真实50ms闭环再次完成，全部22个NPZ数组（7组×轨迹/命令/计数+初态）与修复前基线逐位一致；所有定向事件计数继续吻合。正常仍在24ms越过原下落阈值，未改善运动，也未破坏合法默认行为。

新证据包：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/runs/f10c9354-8b21-4faa-b48c-80fee6a5c157`

逐位回归产物：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/current_closure_identity.json`

修复文件：`/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/sensory_channels.py`

测试：`/run/media/sensen/Data2/cell_wound_prototype/test_sensory_channel_validation.py`

验证工具：`/run/media/sensen/Data2/cell_wound_prototype/verify_current_closure_identity.py`

