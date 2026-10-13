# 已连接感觉通道的选择性切断

**手工研究原型，不是成品。** 规范物理、神经连接、权重、增益保持不变。

## 为什么做这组检查

触觉通道受缺失递质标注阻断，不能继续用该刺激判断整个感觉闭环。当前本体感觉通道与前腿应变通道在图上有到运动池的路径，但图可达不等于功能贡献。

固定50ms，比较正常、切断本体位置/方向/关节角输入、切断campaniform应变输入；切断只是从感觉电流中减去被指定通道原有的分量，不添加运动控制或隐藏驱动。保持原生神经状态与肌肉动力学。重复其中一组检查确定性。报告质心与根高分别测量，不能互相替代。

## 新增可审计链验证

对已保存7组闭环，根据0.5ms emitted运动池事件重新计算完整滤波器、池计数和下一周期命令，70个周期的滤波率误差均为0，命令逐项一致。删除一条运动事件或重复事件都会被测试检出。这里复用同一滤波算法，是事件到命令的完整性检查，不是生物滤波器验证。

- 实现：`/run/media/sensen/Data2/cell_wound_prototype/run_spike_filter_reconstruction.py`
- 测试：`/run/media/sensen/Data2/cell_wound_prototype/test_spike_filter_reconstruction.py`
- 数据：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/spike_filter_reconstruction.json`

## 切断结果

4次50ms真实闭环运行完成，正常组与保存基线及其重复组完全一致，本体切断重复也逐位一致。7项测试通过，父任务另行复跑确认。

| 条件 | 50ms根高 mm | 50ms全身COM高度 mm | 高度90%阈值越过时间 |
|---|---:|---:|---:|
| 正常 | 1.723613 | 1.498134 | 24.0ms |
| 切断claw/hook/hairplate | 1.168585 | 0.636435 | 14.9ms |
| 切断campaniform | 1.723155 | 1.501302 | 23.2ms |

两种切断均在5–10ms周期首先改变运动池计数，10ms开始改变应用命令，15ms保存状态出现差异，符合已有因果顺序。本体切断显著改变此轨迹；应变切断也改变运动放电但终点COM差异很小，不能根据单个高度值宣布改善或无作用。

正常运动放电在15–20ms周期154次，随后降到40–45ms的6次。下一步核对实际感觉电流、输入电导、适应状态与不应期，先分清神经活动衰减原因，不以提高增益掩盖问题。

- 实现：`/run/media/sensen/Data2/cell_wound_prototype/run_connected_feedback_audit.py`
- 测试：`/run/media/sensen/Data2/cell_wound_prototype/test_connected_feedback_audit.py`
- 数据：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/connected_feedback_audit.json`
- 轨迹：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/connected_feedback_traces.npz`
- 已查看图：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/connected_feedback_audit.png`

结论是当前模型中的选择性感觉依赖，不是长期稳定、一般必要性或生物因果证明。模型和增益未变。
