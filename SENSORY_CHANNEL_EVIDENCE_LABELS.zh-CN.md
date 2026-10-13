# 感觉通道证据标注（不改源码，仅文档声明）

**手工研究原型，不是成品。** 本文档**不修改**任何源码或模型；它把每个感觉通道"代码里假设了什么"与"有多少证据"并列，供后续分支引用。之所以不改`sensory_channels.py`：该文件哈希已被既有审计与测试（如`test_sensory_tuning_annotation_audit`）作为断言记录，改动会破坏这些历史断言。

## 通道逐条

| 通道 | 代码中的假设语义 | 实测Verified递质覆盖 | 证据状态 |
|---|---|---|---|
| claw_position | 同亚型共用带符号胫节偏角/速度 | 146/149（98%） | 递质证据强；**统一符号调谐是假设**，文献支持异质偏好角 |
| hook_direction | 同上 | 133/136（98%） | 递质证据强；文献支持方向选择性，但不支持全类同号 |
| club_vibration | 电流恒为0（未实现） | 297/323（92%） | 递质强但**功能未实现**；文献显示单细胞双向调谐 |
| hairplate_angle | 平均腿部关节偏角 | 68/207（33%） | **仅三分之一可进入验证过滤**；文献支持多角度联合编码，不是通用单角度误差 |
| campaniform_strain | "该腿地面接触法向力/负载代理" | 12/60（20%） | **未经验证的假设**：唯一针对果蝇的力学计算认为仅靠地面反力不足以激活行走中的股节钟形场；且仅五分之一有Verified递质 |
| bristle_touch | 接触力范数之和 | 0/2879（0%） | **无Verified递质，结构上无法通过验证过滤**；直接接触角色文献上归给腹侧刚毛而非背面钟形感受器 |
| multidendritic_noxious | 有害刺激 | 0/34（0%） | 无Verified递质，未建模 |

## 后续任何感觉分支必须声明

1. 它针对**哪一组BANC root ID**（含release与哈希）；
2. 该组的**节段身份是否已知**（当前腿部钟形/刚毛行无节段字段，不能挂文献感受器名）；
3. **响应形式**（速率编码、变化率编码、或对向子群）及其**来源是跨物种还是模型**；
4. 它是否改变**已验证递质标签**——不得填补、不得重命名；
5. **对照实验**：至少包含切断/回放，且站立与行走**分别报告**，不拿不存在的站立证据去调参。

## 建议的最小判别实验（尚未实施）

按子代理检索到的已发表驱动系（R34E04-GAL4 / SS69574-Gal4 / SS69602-Gal4）分离跗节与股节钟形感受器来源，作为身份线索；在没有可核验身份来源之前，**不把任何文献感受器名映射到具体root**。

## 相关文件

- 通道覆盖实测：`/run/media/sensen/Data2/cell_wound_prototype/CHANNEL_EVIDENCE_AUDIT.zh-CN.md`
- 文献报告：`/run/media/sensen/Data2/cell_wound_prototype/RESEARCH_LOAD_CONTACT_SENSING.zh-CN.md`
- 转导调谐文献：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/RESEARCH_SENSORY_TUNING.zh-CN.md`
- 本地极性标签审计：`/run/media/sensen/Data2/cell_wound_prototype/SENSORY_TUNING_ANNOTATIONS.zh-CN.md`
- 力矩缺口审计：`/run/media/sensen/Data2/cell_wound_prototype/TORQUE_DEFICIENCY_AUDIT.zh-CN.md`
