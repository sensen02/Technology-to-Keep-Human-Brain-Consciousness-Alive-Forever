# 跨审计一致性核查 + 16个不可驱动肌肉的成因

**手工研究原型，不是成品。** 本文档两部分：先把近期几项审计做一次互相核对（排查"用错episode／用错权重／索引错位"的混淆），再把16个不可驱动肌肉的成因追到原始注释。

## 一、跨审计一致性：12项全部通过

| 检查 | 证据 |
|---|---|
| 体重与既有审计一致 | 本次24.446354755878005 = 既有`support_force_audit`同值 |
| 声明分支用的是同一体重 | 24.446354755878005 |
| 冻结规范XML哈希未变 | `65d3904d…73b6` |
| 声明分支A2归一化 = 体重/6 | 4.0743924593130005（**精确数值**，非格式化字符串） |
| 25μs数组两文件是同一episode | 逐位比较qpos/qvel/act/controls/spike_counts/heights |
| 25μs模型真是25μs | 2.4999999999999998e-05 |
| 普查的16不可驱动 = manifest的16 unresolved = 无池行数 | 16/16/16 |
| 驱动器总数与manifest一致 | 90/90/90 |
| manifest可用数解释差额 | 74 = 90 − 16 |
| 感觉细胞总数跨四项审计一致 | 全部3,788（含逐细胞行数与唯一索引数） |
| 审计细胞属于已发布tier | 全部root命中 |
| 驱动器索引是控制列的排列 | 90个索引 = 0..89 |

**自查发现并修正的两处问题：**
1. 我最初写了一个**空检查**（读取一个根本不存在的字段并硬编码为True）——已替换为真实检查（规范哈希、A2精确值）。
2. 声明分支的A2曾以`%.6f`格式化写成字符串"4.074392"，导致交叉比对出现**假失败**。已改为记录精确数值字段，并重跑该分支。这属于文档精度问题，不是数值错误。

## 二、16个不可驱动肌肉的成因（追到原始注释）

按`(肌肉目标, 腿, 侧)`在BANC 888的efferent注释中匹配，实测覆盖：

| 肌肉目标 | front_leg | middle_leg | hind_leg |
|---|---:|---:|---:|
| tergopleural_promotor_muscle | 8 | **0** | **0** |
| pleural_remotor_and_abductor_muscle | 4 | 1（**仅左侧**） | 2 |
| sternal_adductor_muscle | 2 | 1（**仅右侧**） | 2 |
| pleural_promotor_muscle | **整个数据集0条** | 0 | 0 |

因此16个unresolved分成三类：

1. **6个**：`pleural_promotor_muscle`在**BANC 888里完全不存在**（六条腿各一个actuator）——名称/词表层面的空缺。
2. **8个**：`tergopleural_promotor`只在前腿有注释（左4+右4），中腿与后腿为0。
3. **2个**：中腿该肌肉只有**单侧一个细胞**（RM的remotor只在左侧、LM的adductor只在右侧），因此对侧没有池。

并已确认：这16个actuator在已发布episode中`command_max`**全为0**；7个零放电池`command_max`也全为0；被驱动的67个中没有一个`command_max=0`。

## 三、正确的结论与**不该做**的事

- 这是**注释覆盖缺口**（数据层面），不是池构建逻辑错误：构建器按`(目标,腿,侧)`匹配，缺就如实报告，运行时不替换。
- **不能**把前腿细胞跨腿接到中/后腿同名肌肉上——不同腿由不同运动神经元支配，那样做等于编造连接，违反本项目"禁止臆造连接"的要求。因此我不实施"镜像映射"。
- 可行路径只有：(a) 找到**腿分辨**的注释来源补这些池；(b) 或在报告与结论中明确限定"16/90不可控"，不据此断言整套六足控制的能力上限。

## 四、产物与边界

- 脚本：`/run/media/sensen/Data2/cell_wound_prototype/run_cross_audit_consistency.py`
- 测试：`/run/media/sensen/Data2/cell_wound_prototype/test_cross_audit_consistency.py`（4项通过，含"不得有空检查"的断言）
- 结果：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/cross_audit_consistency/cross_audit_consistency.json`

边界：一致性核查只证明各项审计之间的量值/来源互相吻合，**不等于**任何一项结论因此变成生物学验证；注释覆盖统计来自BANC 888 release的`efferent`子集（1,033条），不含其他数据版本。
