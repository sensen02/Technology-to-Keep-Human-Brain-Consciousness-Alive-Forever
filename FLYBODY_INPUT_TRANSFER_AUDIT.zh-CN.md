# 感觉输入到膜电位的分层审计

**手工研究原型，不是成品。未验证连接组传播、运动输出或站立。**

## 本轮实施

- `/run/media/sensen/Data2/cell_wound_prototype/run_flybody_loop.py`：episode 现在记录 `actual_network_input_by_channel`，按腿/通道输出实际网络入口电流的细胞数、和、最小、最大值；位置在外部触觉、club注入、符号/消融、sensory_cut之后。不是仅记录 encoder 的原始描述。另记录 sensory_roles。
- `/run/media/sensen/Data2/cell_wound_prototype/run_flybody_input_transfer_audit.py`：用 published FlyBody 模型、经过 manifest/identity 哈希验证的12,000神经元身份和3,788感觉根分组，做六腿×两输入共12次扰动。
- `/run/media/sensen/Data2/cell_wound_prototype/test_flybody_sensory_wiring.py`：增加实际电流摘要测试，包括空分组和切断后的零向量。

## 测量结果

沿用现有 ChannelConfig，没有调增益：

| 单腿扰动 | 目标通道 | 每细胞电流增量 | 0.5ms后平均膜电位增量 |
|---|---|---:|---:|
| coxa位置 +0.125 rad，诊断角色 hairplate=coxa | hairplate_angle | +0.125 nA | +3.0487804878 mV |
| tibia速度 +1.234 rad/s，角色 claw=tibia | hook_direction | +0.03702 nA | +0.9029268293 mV |

六腿均通过同样的电流及膜电位检查。目标通道以外膜电位变化为0；单步目标脉冲均为0。**不能把亚阈值膜响应称为脉冲传播。**

## 严格局限

膜方程测试仅隔离已测得的目标通道电流增量，使用没有设置连接的 ConductanceNetwork。这样可验证根索引、电流路由与膜输入接受，不混入递归传播，但**不是完整 BANC 连接组实验**。没有加入虚构连接，也没有向motor pool直接注流。

观测扰动可能同时改变接触等其他感觉量，故电流隔离只用于明确测试目标通道；完整观测电流摘要保留在JSON中。all-coxa仍是未验证诊断映射，不能用于声称所有hairplate生物学正确。

新 episode 记录代码已加入，但本轮未运行完整 episode，因此尚无带该新字段的闭环轨迹。历史结果不重写。下一步应在真实连接组上进行匹配初态、有/无该输入的配对实验，分开报告感觉脉冲、下游膜响应及运动脉冲。

## 验证与交付

12个分层扰动通过；FlyBody相关联合测试 **62 tests / OK**。回归里旧artifact断言不代表重算。上一轮覆盖图已实际查看，本轮没有新的图形输出。

结果：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/flybody_loop/flybody_input_transfer_audit.json`

桌面副本目录：`/home/sensen/Desktop/cell_wound_prototype/input_transfer_audit/`
