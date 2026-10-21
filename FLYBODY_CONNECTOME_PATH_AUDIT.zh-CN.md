# 运动池零脉冲：结构与亚阈值诊断

**手工研究原型，不是成品。没有调整输入、增益、连接或阈值；不是站立验证。**

## 实验保持不变

重复上一轮三组匹配初态实验：真实重建BANC tier的12,000根顺序与存档一致，3,788感觉细胞，202运动池细胞；现有We/Wi×0.2；零背景；0.5ms步长；20ms左前腿诊断电流脉冲，观察100ms。脉冲总数与上一轮完全相同。

新增诊断使用网络实际矩阵，方向严格为 **W[target, source]**（与step中的W@arrivals一致）。合并兴奋和抑制边作有向最短路径搜索，因此这里只是无符号结构可达性，不是兴奋可驱动性。

## 结果

| 指标 | hairplate位置代理输入 | hook速度输入 |
|---|---:|---:|
| 无符号可达运动池细胞 | 202/202 | 202/202 |
| 实际发放非感觉细胞数 | 3 | 17 |
| 非感觉脉冲总数 | 4 | 17 |
| 已发放源到运动池的边记录数（包含感觉源） | 11 | 92 |
| 收到非零兴奋电导的运动细胞 | 3 | 28 |
| 收到非零抑制电导的运动细胞 | 8 | 34 |
| 运动池最高膜电位 | −52.088081909 mV | −51.039887175 mV |
| 距−50mV阈值的最小间隔 | 2.088081909 mV | 1.039887175 mV |
| 运动池脉冲 | 0 | 0 |

兴奋/抑制细胞集合可能重叠，不能直接相加成唯一细胞数。边记录数不是神经元数，也不只来自非感觉细胞。所有发放非感觉根、源至运动池边和202个运动细胞的逐细胞诊断均保存在JSON。

零输入对照全程无脉冲、无电导事件，运动池保持−65mV。

## 能支持什么

当前tier中不存在“这两个输入群完全没有到运动池的有向结构路径”这一情况，而且部分运动细胞确实收到事件并去极化。然而大量结构路径没有在本次输入下实现脉冲传播；结构可达性不能替代动力学证据。不能把零运动脉冲简单归因为接线断开，也不能宣称只差某个幅度就能实现闭环或稳定。

电压峰值在每步结束读取，含初始静息电位；本次运动细胞均未发放，故没有重置掩盖阈值的问题。电导峰值在step内指数衰减后读取，明确命名post_decay，不冒充到达瞬间峰值。此表不展示纯抑制细胞的最低电位，因为最大值含初始静息。

## 验证与文件

新增方向单元测试用0→1→2有向链验证矩阵方向及不可达/空源。新增artifact核对发放根计数、运动池计数、电导非负和没有直接向motor注流。与现有配对输入及接线测试联合 **9 tests / OK**。

已实际查看 `/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/flybody_loop/flybody_motor_subthreshold.png`：三组电位峰值排序曲线、−50mV阈值及原型标签清晰，全部曲线低于阈值。各组独立排序，横轴不是同一细胞对齐比较。

数据：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/flybody_loop/flybody_connectome_path_audit.json`

传播图：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/flybody_loop/flybody_connectome_path_audit.png`

电位图：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/flybody_loop/flybody_motor_subthreshold.png`

诊断源码：`/run/media/sensen/Data2/cell_wound_prototype/tools_connectome_path_diagnostics.py`

运行源码：`/run/media/sensen/Data2/cell_wound_prototype/run_flybody_connectome_transfer_audit.py`

测试：`/run/media/sensen/Data2/cell_wound_prototype/test_connectome_path_diagnostics.py`

本轮输出独立path_audit文件，不覆盖上一轮transfer_audit数据和图。当前运行程序生成新的path_audit输出。

下一步应给实际发放的非感觉根连接原始注释，辨认中间通路，并对最接近阈值的运动根进行事件时序审计；不以增益搜索取代证据。
