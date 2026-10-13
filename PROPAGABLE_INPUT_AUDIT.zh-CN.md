# 可传播感觉输入审计：只有7.24%的注入电流能传出去

**手工研究原型，不是成品。** 只读既有已发布run的保存状态，不积分、不改参数、不改增益。

## 一、方法与自检

把`run_sensory_frontend_balance.py`记录的**逐细胞**平均注入电流，与BANC的`neurotransmitter_verified`按root连接：`claw/hook/...`按闭包真实使用的tier细胞。**可传播**定义为"注入到Verified NT非空细胞上的电流"——因为闭包的突触过滤作用于已验证的前突触标签，未标注细胞即使放电也传不出去。

自检：我第一次实现把`nan`当成了有效标签，结果"3,788/3,788全部Verified"，与独立审计（刚毛0/2879）矛盾，因此判定为bug并修正。现在脚本内置**与逐通道覆盖审计的交叉断言**：claw 146、hook 133、club 297、hairplate 68、campaniform 12、bristle 0、多树突 0，全部一致才允许写出结果。

## 二、结果

| 通道 | 注入电流 nA | 可传播 nA | 通道内可传播比例 | Verified细胞 |
|---|---:|---:|---:|---:|
| bristle_touch | **775.52** | **0.00** | **0%** | 0/2879 |
| campaniform_strain | 43.49 | 9.67 | 22% | 12/60 |
| hook_direction | 30.63 | 29.84 | **97%** | 133/136 |
| claw_position | 21.60 | 20.70 | **96%** | 146/149 |
| hairplate_angle | 10.23 | 3.64 | 36% | 68/207 |
| club_vibration | 0.00 | 0.00 | — | 297/323（但通道值为0） |
| multidendritic_noxious | 0.00 | 0.00 | — | 0/34 |

**合计：注入881.47 nA → 可传播63.86 nA，占7.24%**；3,788个感觉细胞中只有656个带Verified标签。

## 三、含义

1. **92.8%的感觉输入电流被浪费**：主要因为在2,879个无Verified递质的刚毛细胞上注入775.5 nA，而它们的突触被验证过滤移除。
2. 真正进入网络的驱动排序是 **hook 29.8 > claw 20.7 > campaniform 9.7 > hairplate 3.6 > bristle 0**——与"证据强度"排序一致，也就是说**目前闭环实际依靠的是位置/方向类本体感觉**，这与其92–98%的递质覆盖相符。
3. `club_vibration`虽然297/323有Verified标签，但通道值恒为0（未建模），因此贡献0；不能把它算作参与闭环。
4. 结合前几轮：初始站姿支撑63.8%（右中腿悬空）、t=0位置类通道恒为0（参考=初态）、t=0力矩覆盖26%、约95–97ms趴地、最大实际力矩>最大所需——**瓶颈链条是"参考零点+归一化缺少测量依据"与"命令模式/时序"，而不是力量或细胞数量**。

## 四、下一步（仍未实施）

1. 明确声明参考零点的选择（姿态目标 vs 解剖零点），并写入文档。
2. 任何"输入强度"结论都必须用**可传播电流**而非注入电流表述。
3. 若要恢复触觉功能，需要独立的递质证据来源或**单独的显式类别先验表**，绝不改写Verified字段。
4. 下一步应优先改**参考零点与归一化**并做切断对照，而不是调运动增益或加CPG/PD。

## 五、产物与边界

- 脚本：`/run/media/sensen/Data2/cell_wound_prototype/run_propagable_input_audit.py`
- 测试：`/run/media/sensen/Data2/cell_wound_prototype/test_propagable_input_audit.py`（5项通过，含"nan不得被当作标签"的回归）
- 结果：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/propagable_input/propagable_input.json`
- 图：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/propagable_input/propagable_input.png`
- 上游逐细胞电流：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/sensory_frontend_balance/sensory_frontend_balance_traces.npz`

边界：单一run、单一姿态、单一tier；"可传播"是数据层面的上界（有标签≠放电到达），不含任何极性/调谐推断。
