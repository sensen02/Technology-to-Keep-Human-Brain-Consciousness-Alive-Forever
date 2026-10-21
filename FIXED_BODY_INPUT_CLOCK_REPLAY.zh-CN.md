# 固定真实身体输入跨神经步长传播

**手工研究原型，不是成品。诊断重放，不控制身体，不作为闭环成功。**

固定已保存派生初态基础正常臂100ms身体输入，每5ms保持相同电流；通过逐通道count/min/max/sum拒绝非均匀或重叠输入，重建12000维电流。实际BANC tier身份一致，零背景，连接与参数固定。全体脉冲事件、运动膜电位与解码结果保存。

## 核验与结果

11项回归通过。基础0.5ms重放的每个神经元、每个5ms窗口计数与原身体实验**精确相等**，实际应用命令也精确相等，证明固定输入重建有效。源码快照核验通过；实际看图确认运动总趋势接近但末端解码差异显著、最大控制长时间裁剪。

|神经步长|全网脉冲|感觉脉冲|运动脉冲|最早运动脉冲|
|---|---:|---:|---:|---:|
|0.5ms|10331|2435|534|10ms|
|0.25ms|10283|2508|539|9.5ms|
|0.125ms|10292|2523|538|9.375ms|
|0.0625ms|10268|2547|536|9.3125ms|

相对基础，首个变化是感觉根720575941354084144（局部43）：基础6ms，半步5.75ms，四分/八分步5.625ms。首个运动根差异是720575941493231536（局部3532）：基础10ms、其他9.5/9.375/9.3125ms。

改变精确事件的根数2483/2529/2541；这包含仅时间网格差异，不能直接称这些根功能改变。运动5ms窗口计数首次差异发生在5—10ms窗口。相对基础逐执行器命令最大差可达1，即裁剪后的全控制范围；总运动脉冲接近不能替代逐池/逐时刻收敛。

## 解释

**在完全相同身体输入下，神经传播与解码仍有差异，因此此前完整身体分叉不只来自身体步长或不同感觉历史。**但尚未量化积分、不应期、阈值取样、递归传播和滤波各自贡献，不声称某项单独主导。

基础最大控制曲线长期1可能掩盖某些执行器差异；查看最大值不能判定逐池一致。膜电位是重置后记录，不含阈值前峰值。

下一步：用本实验全体事件，匹配首个运动根的真实到达边/时刻与膜轨迹，固定上游事件分别重放以隔离单目标积分与递归传播；独立数值分支必须通过夹具、事件延迟和跨步长回归，不能通过改阈值/增益/符号让站立通过。

## 成果路径

- 入口：`/run/media/sensen/Data2/cell_wound_prototype/run_fixed_body_input_clock_replay.py`
- 测试：`/run/media/sensen/Data2/cell_wound_prototype/test_fixed_body_input_clock_replay.py`
- 结果：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/fixed_body_input_clock_replay/20261013_112901/clock_replay_results.json`
- 全体事件及电流：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/fixed_body_input_clock_replay/20261013_112901/clock_replay_traces.npz`
- 图：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/fixed_body_input_clock_replay/20261013_112901/clock_replay.png`
- 源码快照：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/fixed_body_input_clock_replay/20261013_112901/source_snapshot.zip`
