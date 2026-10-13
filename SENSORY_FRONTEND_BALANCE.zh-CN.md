# 感觉前端驱动平衡审计：88%的输入电流进了"死胡同"

**手工研究原型，不是成品。** 只读既有25μs正常分支的保存状态，复用真实编码路径（`observe_quantities` → `ChannelSensoryEncoder`），不积分、不改参数、不重写电流。

## 一、结构性事实：参考姿态就是初始姿态

`observe_quantities(model, data, reference=seated, ...)`以**闭包的seated初态本身**作参考，因此

- t=0时`claw_position`、`hook_position`、`hairplate_angle`的偏差**恰好为0**（实测：这三个通道第0个样本全零）；
- 网络感知的是"相对起始姿态的偏离"，**不是可追溯的解剖零点**。这与本项目文献研究的要求（零点必须是解剖坐标，不是姿态目标）不一致；
- 直接后果：**在一个支撑力只有体重63.8%、右中腿悬空的姿态下，位置类感觉报告"没有误差"**。

## 二、驱动去向（0–200ms，2001个样本累计）

| 通道 | 细胞数 | Verified覆盖 | 归一化最大值 | 输入电流占比 |
|---|---:|---:|---:|---:|
| bristle_touch | 2879 | **0%** | 1.00 | **88.0%** |
| campaniform_strain | 60 | **20%** | **6.85** | 4.9% |
| hook_direction | 136 | 98% | 3.50 | 3.5% |
| claw_position | 149 | 98% | 1.70 | 2.4% |
| hairplate_angle | 207 | 33% | 0.37 | 1.2% |
| multidendritic_noxious | 34 | 0% | 0.00 | 0.0% |
| club_vibration | 323 | 92% | **0.00** | 0.0% |

两个要点：

1. **88%的输入电流注入2,879个没有任何Verified递质的细胞**，其突触在验证过滤下无法传播——即绝大部分前端驱动进入死胡同。真正能到达网络的只有约12%。
2. **通道增益不在同一量纲**：接触力以`reference_force=1.0`归一，实测单腿接触力范数最大**6.85**（单腿体重份额为4.07）；速度以10 rad/s归一得到3.50。配置里已明确声明这些normaliser是"项目既有声明值，不是测量值"。
3. **t=0时只有两个通道非零**：`bristle_touch`与`campaniform_strain`——正好是证据最弱的两个（0%与20%）；而证据最强（92–98%）的三个通道此刻恰好为零；`club_vibration`与多树突**全程精确为0**（未建模）。

## 三、与之前审计的关系

- 上轮力矩审计已表明：瓶颈是命令的模式与时序、不是力量上限。本轮补充前端原因：**早期唯一可传播的输入来自证据最弱的通道，且其量级未按测量标定**。
- 这不是"感觉细胞太少"，也不是"力量不够"；是**参考零点与归一化都缺少测量依据**。

## 四、可执行的下一步（尚未实施，且不靠调参凑站立）

1. **明确声明参考零点是姿态目标**，或改为可追溯解剖零点——两者必须选一个并写入文档，不能含糊。
2. **分别报告"注入电流"与"可传播电流"**：按通道统计注入到有Verified递质细胞上的部分，避免把死胡同电流当成有效驱动。
3. 若要做接触/负载分支，先把`campaniform`的语义改为显式假设并做切断对照，不据站立表现调normaliser。
4. `club_vibration`与多树突目前全程0，属已声明的未建模通道，不应在结论中当作"参与了闭环"。

## 五、产物与边界

- 脚本：`/run/media/sensen/Data2/cell_wound_prototype/run_sensory_frontend_balance.py`
- 测试：`/run/media/sensen/Data2/cell_wound_prototype/test_sensory_frontend_balance.py`（5项通过）
- 结果：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/sensory_frontend_balance/sensory_frontend_balance.json`
- 数组：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/sensory_frontend_balance/sensory_frontend_balance_traces.npz`
- 图：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/sensory_frontend_balance/sensory_frontend_balance.png`

边界：单一已发布run、单一姿态、单一tier；电流占比是注入量而非可传播量（后者上界为"仅Verified细胞"那一列）；血统上normaliser无测量依据，故任何"某个感觉通道足够/不足"的结论都不能从本审计单方面推出。
