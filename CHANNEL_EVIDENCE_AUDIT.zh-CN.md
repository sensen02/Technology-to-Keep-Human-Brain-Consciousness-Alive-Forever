# 感觉通道证据审计：接触/负载恰恰是证据最弱的

**手工研究原型，不是成品。** 本报告只读已发布run与BANC数据，不填补任何递质标签、不推断极性。

## 一、本轮自己实测：逐通道递质证据覆盖率

对已发布run（`add80e12-4b1b-45fc-90be-2336ddf7757c`）实际使用的全部3,788个感觉通道细胞，逐个按`banc_888_id`连接BANC 888的`neurotransmitter_verified`字段（4项测试通过，图已查看）：

| 通道 | 细胞数 | 有Verified NT | 覆盖率 |
|---|---:|---:|---:|
| claw_position（爪位置） | 149 | 146 | **98.0%** |
| hook_direction（钩方向） | 136 | 133 | **97.8%** |
| club_vibration（棒振动） | 323 | 297 | **92.0%** |
| hairplate_angle（毛板角度） | 207 | 68 | **32.9%** |
| campaniform_strain（钟形感受器应变） | 60 | 12 | **20.0%** |
| multidendritic_noxious（多树突有害） | 34 | 0 | **0%** |
| bristle_touch（刚毛触觉） | 2879 | 0 | **0%** |

有Verified标签的细胞**全部是acetylcholine**，没有出现第二种；48/60钟形、139/207毛板、2879/2879刚毛、34/34多树突只存在**预测**标签，未被提升为Verified。

含义：闭环的突触过滤作用于"已验证递质的前突触标签"，因此**通道大小与能否驱动网络无关**——2,879个刚毛细胞贡献0，而60个钟形细胞只有12个可能进入过滤。**步行最需要的接触/负载信息，正是证据最弱的一环。**

## 二、文献结论（子代理有界检索，6篇）与本地判定的关系

- 唯一针对果蝇的力学计算认为：前向行走中**仅靠地面反力不足以激活股节钟形感受器场**（Goldsmith et al. 2025, J R Soc Interface 22(226):20240559, [PMC12056673](https://pmc.ncbi.nlm.nih.gov/articles/PMC12056673/)；最大等效力矩对应46.019µN，受体位移低于NompC的7Å热阈值）。
- 因此本地把`campaniform_strain`当作"该腿地面接触法向力/负载代理"是**未经验证的假设**，不是文献事实；结合它只有20%的Verified覆盖，本项目的处理应当是**保留为显式标注的假设分支**。
- 最强行为学证据是**预印本**（Custódio et al. 2026, bioRxiv），且其**站立**实验为阴性（站立果蝇地面间隙无变化）；该证据是类别级、无单感受器身份，不可外推到单腿。
- 跗节钟形感受器位于**背面**、跗分节与爪附近，不在接触面上；直接接触被归给腹侧机械感觉刚毛（Zhang et al. 2025, PLoS Genet 21(12):e1011980, [PMC12768414](https://pmc.ncbi.nlm.nih.gov/articles/PMC12768414/)）。
- 没有任何来源把某条腿的钟形感受器映射到BANC/FlyWire root；Kuan et al. 2025 ([PMC12048489](https://pmc.ncbi.nlm.nih.gov/articles/PMC12048489/)) 指出腿部感觉轴突是最难重建的一类，支持"BANC腿部感觉身份不可靠"的判断。

## 三、这改变什么

1. **不能靠补钟形/刚毛通道来救站立**：既无极性映射，也无递质证据，且站立方向的文献本身是阴性或未做。
2. **证据最强的通道是本体感觉类**（爪98%、钩98%、棒92%），它们编码位置/方向/振动——这也解释了为什么现有闭环里起作用的是位置型信号。
3. **下一步合理做法**是：把钟形通道明确标注为假设并做对照实验，或按子代理建议用已发表的钟形亚群驱动系(R34E04-GAL4 / SS69574 / SS69602)去分离跗节与股节来源——这需要一个可核验的身份来源，而不是猜。
4. 结合上一轮力矩审计（崩塌时最缺髋部侧向控制、t=0胫节无支撑、最大实际力矩>最大所需），**当前瓶颈是命令的模式与时序，不是力量、也不是"某一类感觉细胞太少"**。

## 四、产物

- 脚本：`/run/media/sensen/Data2/cell_wound_prototype/run_channel_transmitter_audit.py`
- 测试：`/run/media/sensen/Data2/cell_wound_prototype/test_channel_transmitter_audit.py`（4项通过）
- 结果：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/channel_transmitter_audit/channel_transmitter_audit.json`
- 图：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/channel_transmitter_audit/channel_transmitter_audit.png`
- 文献报告：`/run/media/sensen/Data2/cell_wound_prototype/RESEARCH_LOAD_CONTACT_SENSING.zh-CN.md`
- 文献账本：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/research_load_contact_sources.json`

## 五、边界

覆盖计数只针对该run的tier；空标签表示"缺少经验证标签"，不等于"无递质"；本审计不改变任何生理数值、不重命名通道、不把预测当作验证；不含任何极性/调谐/节段身份推断。
