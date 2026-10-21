# 运动细胞事件时序与原始注释

**手工研究原型，不是成品。无参数调整、无身体控制，不是站立验证。**

## 新证据

在同一真实连接组、相同输入和初态下，记录所有发放根及发放时间，并只读检查网络step将消费的延迟环槽。对每组电位峰值最高的5个运动细胞，输出逐步兴奋/抑制到达增量及膜电位。

40个相关根以 banc_888_id 唯一匹配v888原始注释，包含实际发放的非感觉根、重点运动根及其实际发放上游源。未用类型名替换根身份，未由注释重分配符号。

## 最接近阈值的运动根

两种输入对应同一个根 **720575941483515389**：

- cell_type：trochanter_extensor_LinJ
- super_class：motor；cell_sub_class：front_leg_motor_neuron
- side：left；body_part_effector：front_leg
- cell_function_detailed：extend_coxa_trochanter_joint
- manc_cell_type：Sternotrochanter MN

| 输入 | 兴奋到达（区间开始ms） | 抑制到达（区间开始ms） | 电位峰值及区间结束时间 |
|---|---|---|---|
| hairplate位置代理 | 5、10、15、20 | 无 | −52.088081909mV，23.5ms |
| hook速度 | 13、17 | 17、17.5、18 | −51.039887175mV，18ms |

峰值均未达到模型−50mV阈值。事件增量不是持续电导：到达后仍有电导衰减和膜积分，因此峰值可晚于最后一次兴奋事件。hook输入出现兴奋和抑制时序重叠，但没有做去抑制反事实实验，不能断言抑制是未发放的唯一原因。

## 非感觉细胞注释

hairplate输入发放3个非感觉细胞：2个AN17A008（ascending）及1个IN13B001（ventral_nerve_cord_intrinsic）。hook输入发放17个非感觉细胞：2个ascending（AN05B104、AN12B001）及15个ventral_nerve_cord_intrinsic。各根和原始字段在JSON中保留；这些标签不等于已验证的因果功能。

值得单独复核：重点运动根的 neurotransmitter_predicted 为 gaba，neurotransmitter_verified 缺失。这里只报告预测字段，**不把它当实验证实，不据此改递质或边符号**；该根自身本次没有发放，因此其输出递质也不能直接解释本次它为何未发放。

## 验证

新增测试从每个上游源的实际发放时间，按现有2ms延迟和真实边权重独立重建每个重点运动根每步到达增量，与记录逐项核对。另核对总事件数、逐根脉冲、电压峰值和注释唯一连接。联合相关测试 **11 tests / OK**。

已实际查看 `/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/flybody_loop/flybody_focus_motor_events.png`：两组重点根相同，电位曲线均低于阈值，到达事件时序与表一致，标签和原型说明清晰。

神经模型参数本身是未校准点神经元约化，不能把此阈值间隔当作果蝇生理测量。程序对私有延迟环仅作只读诊断，未修改冻结核心；未来核心布局改变时需重新验证接口。

## 输出

原始事件：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/flybody_loop/flybody_connectome_event_audit.json`

注释与摘要：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/flybody_loop/flybody_connectome_event_annotations.json`

时序图：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/flybody_loop/flybody_focus_motor_events.png`

运行程序：`/run/media/sensen/Data2/cell_wound_prototype/run_flybody_connectome_transfer_audit.py`

注释程序：`/run/media/sensen/Data2/cell_wound_prototype/tools_connectome_event_annotation.py`

测试：`/run/media/sensen/Data2/cell_wound_prototype/test_connectome_event_audit.py`

桌面交付目录：`/home/sensen/Desktop/cell_wound_prototype/connectome_event_audit/`

下一步应核查实际边的递质符号来源及预测/验证覆盖，并评估在不更改证据约束的前提下如何回到真实身体闭环。不能把递质预测、近阈值或有向可达性当作已校准闭环。
