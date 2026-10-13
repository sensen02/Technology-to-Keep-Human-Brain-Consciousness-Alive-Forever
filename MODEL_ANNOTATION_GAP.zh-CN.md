# 模型与注释的双向结构缺口

**手工研究原型，不是成品。** 只读比对：模型能驱动什么 vs 注释命名了什么。**不做任何替换、镜像或猜测。**

## 一、可驱动的自由度

规范模型的每条腿只有 **7 个关节**：

| 关节 | 数量/腿 |
|---|---:|
| Coxa_yaw / Coxa_pitch / Coxa_roll | 3 |
| Trochanter_yaw / Trochanter_pitch / Trochanter_roll | 3 |
| Tibia_pitch | 1 |

**没有股节（femur）关节，也没有跗节（tarsus）关节。** 90个actuator来自12个不同肌肉名（对应15个带后缀的细分名）。

## 二、反向缺口：注释里有、模型里没有

| 注释中的肌肉 | 注释到的腿部运动神经元数（全腿合计） |
|---|---:|
| accessory_tibia_flexor_muscle | 71 |
| long_tendon_muscle | 48 |
| femur_reductor_muscle | 34 |
| tarsus_depressor_muscle | 23 |
| tarsus_levator_muscle | 7 |

这5类肌肉在BANC里有腿分辨的注释细胞，但模型**没有任何actuator或关节**可以驱动它们——尤其跗节抬/压肌与股节内收肌，恰恰与**脚部摆放和接触**有关。

## 三、正向缺口：模型里有、注释在对应腿上没有

16个actuator（占90的18%）在其自身腿/侧找不到注释池：

| 原因 | 数量 |
|---|---:|
| 该目标在整个数据集中不存在（`pleural_promotor_muscle`，六条腿各一） | 6 |
| 该目标只在其它腿有注释（`tergopleural_promotor` 仅前腿8个） | 8 |
| 该目标在中腿只有对侧一个细胞（RM的remotor仅左侧、LM的adductor仅右侧） | 2 |

并且已确认这16个actuator在已发布episode中`command_max`严格为0。

## 四、结论与**禁止**的做法

- 两个缺口都是**数据/模型覆盖**问题，不是匹配逻辑错误；运行时按`(目标,腿,侧)`匹配、缺失即如实报告。
- **不得**把前腿细胞跨腿接到中/后腿同名肌肉，也不得用对侧细胞补单侧缺口——不同腿/侧由不同运动神经元支配，那样等于编造连接。
- 因此在这些缺口补上之前：任何六足对称控制任务的结论都必须限定"16/90不可控、且模型无跗节/股节自由度"；与"脚部接触与放置"有关的行走能力，在当前模型里**连执行器都不存在**。

## 五、产物与边界

- 模型侧脚本：`/run/media/sensen/Data2/cell_wound_prototype/run_model_annotation_gap_audit.py`
- 注释侧导出（需pandas解释器）：`/run/media/sensen/Data2/cell_wound_prototype/tools_export_banc_leg_muscle_coverage.py`
- 测试：`/run/media/sensen/Data2/cell_wound_prototype/test_model_annotation_gap_audit.py`（6项通过，含"不得提出替换方案"的键名检查）
- 结果：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/model_annotation_gap/model_annotation_gap.json`
- 覆盖导出：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/model_annotation_gap/banc_leg_muscle_coverage.json`
- 图：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/model_annotation_gap/model_annotation_gap.png`

**自查记录**：第一版我用正则自造肌肉名解析，把tibia_flexor等已建模肌肉误列为"反向缺口"（10项而非5项）；改用项目自己的`parse_actuator`+`ALIASES`后修正为5项，并断言全部90个actuator名都能解析、无缺别名。

边界：单一注释release与单一模型版本；覆盖只统计efferent子集（腿部396条）；不对动物本身是否有该肌肉作任何断言。
