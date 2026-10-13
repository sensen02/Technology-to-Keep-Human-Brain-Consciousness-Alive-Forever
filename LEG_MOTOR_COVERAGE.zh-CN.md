# 缺口性质判定：是**注释覆盖缺口**，不是命名问题

**手工研究原型，不是成品。** 只读审计，未修改任何池或清单。

## 一、为什么必须先做这个判定

如果缺失的池只是"换了名字"，那我可以直接补；如果数据里根本没有，那就只能要新数据。因此本轮不只看`peripheral_target_type`一个字段，而是搜遍**整个注释词表**和**按神经分类的efferent群体**。

## 二、实测（BANC 888，`efferent`子集1,033条）

按腿神经分类的efferent细胞：

| 神经段 | 细胞数 |
|---|---:|
| prothoracic（前腿） | 86 |
| mesothoracic（中腿） | **124** |
| metathoracic（后腿） | **125** |
| 合计 | 335 |

其中在腿神经上出现**57种不同cell_type**——说明中/后腿的运动神经元注释**并不稀疏**，是**特定肌肉组**缺失。

## 三、四个缺口目标的逐项核对

| 目标 | 腿神经上的细胞 | 分布 |
|---|---:|---|
| pleural_promotor_muscle | **0** | 无 |
| tergopleural_promotor_muscle | **0** | 那8个细胞在**背侧前胸神经**上，不在腿神经上 |
| pleural_remotor_and_abductor_muscle | 3 | T2左、T3左、T3右（**无T2右**） |
| sternal_adductor_muscle | 2 | T1左、**T2右**（**无T2左**） |

**全词表搜索**：所有细胞类型名含"promotor"的细胞**全部**位于前胸（前腿）神经——`left/right_dorsal_prothoracic_nerve`，中/后腿神经上**一个都没有**。

**结论**：这是**注释覆盖缺口**，不是我们的命名映射问题。**16个中的14个在本release里根本无法补齐**（只能从更新release或补充表得到）。

## 四、一个**由注释自身命名支持**的候选（仅记录，未接线）

注释里存在一个复合细胞类型 **`tergopleural_promotor_pleural_promotor`**（含`_miller_28_30`变体），共8个细胞，全部位于前胸背侧神经。名称本身表明**同一组运动神经元同时支配tergopleural与pleural promotor**。

因此**前腿**的两个`pleural_promotor` actuator（LF、RF）可以合理地与已分配给`tergopleural_promotor`的池共享。这属于"注释命名所支持的读法"，不是猜测。

**但我没有实施**：它会让前腿出现三方共享池（a、b与pleural_promotor同池），必须声明并配对照；而且它只解决**16个中的2个**。

## 五、因此对新数据的具体要求（与派出的检索任务一致）

1. **中/后腿的 promotor 注释**（tergopleural 与 pleural promoter），或更新的BANC release／论文补充表；
2. **T2右侧 pleural_remotor_and_abductor** 与 **T2左侧 sternal_adductor** 各1个细胞级别的注释；
3. 若有中/后腿 tergopleural promotor，还会同时解除 LM/LH/RM/RH 各2个 = 8个actuator。

## 六、产物与边界

- 脚本：`/run/media/sensen/Data2/cell_wound_prototype/run_leg_motor_coverage_audit.py`
- 测试：`/run/media/sensen/Data2/cell_wound_prototype/test_leg_motor_coverage_audit.py`（4项通过，含"候选不得已接线"的断言）
- 结果：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/leg_motor_coverage/leg_motor_coverage.json`
- 相关本地文件（前几轮生成，本轮首次核对）：`/run/media/sensen/Data2/cell_wound_prototype/data/banc888/design_pool_table.json`、`promotor_adductor_pools.json`

边界：单一release；"未注释"不等于"动物没有该神经元"；共享池属未经验证的募集假设，使用前必须声明并做切断/回放对照。
