# 钟形通道成分审计：**身份证据与递质证据不重叠**

**手工研究原型，不是成品。** 只读审计，未填补任何标签、未改变通道语义。

## 一、部署通道里那60个钟形细胞是什么

| cell_type | 细胞数 | 有Verified递质 | 节段身份 |
|---|---:|---:|---|
| CS（泛类标签） | 23 | **9** | 无 |
| SNcs01f | 12 | 0 | 无 |
| **SNpp53** | **11** | **0** | **有：trochanter（FBbt:00048059）** |
| bCS（泛类标签） | 4 | 1 | 无 |
| SNpp45 | 3 | 0 | 无（且外部标注冲突） |
| SNpp52 | 2 | 0 | 无 |
| SNxx30 | 2 | 0 | 无（且外部标注冲突） |
| SNpp43 | 2 | 1 | 无（且外部标注冲突） |
| SNpp39 | 1 | 1 | 无（且外部标注冲突） |
| 合计 | **60** | **12** | 11 |

## 二、核心结果：两类证据**完全不相交**

| 量 | 值 |
|---|---:|
| 能给出节段身份 **且** 有Verified递质 | **0** |
| 有Verified递质但无节段身份 | **12** |
| 有节段身份（trochanter）但无Verified递质 | **11** |

也就是说：**能确定"这是哪一节段的感受器"的11个细胞，一个都不能传播信号；而能传播的12个细胞，标注只到泛类"CS/bCS"（其中2个还被外部标注否证）。**

## 三、另有8个细胞的"钟形"标签被外部否证

`SNpp39 / SNpp45 / SNpp43 / SNxx30` 共8个细胞，在BANC里记为campaniform，但在外部本体（VFB）里分别被标注为股节弦音器官club神经元(FBbt:00049556)、adult SNta23、adult SNxx33、未表征的SNxx30。**其中2个还是有Verified递质的**。这些行目前仍被部署通道当作campaniform使用。

## 四、因此**不能**声称的（必须写进模型假设）

- 不能给任何行挂上具体感受器名（如Sc3/TrFa/TrFp/TrG）；
- 不能把任何行称为femoral/tibial/tarsal（只有11个有节段，且是trochanter）；
- 不能声称这是**负载/力编码器**——唯一针对果蝇的力学计算是**阴性**的（股节场位移0.71–5.73Å，全部低于NompC的7Å阈值）；
- 不能声称存在**脚部接触通道**——没有任何来源测过果蝇的接触变量；跗节钟形感受器位于**背面**，且跗分节处/爪附近，而**跗节/股节没有关节也没有肌肉**；
- 不得填补任何递质标签。

## 五、与子代理检索的衔接

- **P5已解决**：Dinges 2021并非无法获取——科隆大学仓库有作者接受稿（HTTP 200），逐节段SEM计数到手（femur 10–12+1、trochanter 7–8+4–5+3、tibia 2+2–3、tarsus 2+2+3–4+1，**coxа上没有钟形感受器**）。
- **驱动系无细胞清单**：CS1/CS2/CS3只被笼统描述，且论文内部存在`R34E04`与`R34E03`命名不一致；`SS69574`/`SS69602`在VFB与split-GAL4资源里零命中。
- 唯一"已发表方法模板"是翅膀（Lesser/Tuthill 2025）与腿部毛板（Pratt 2026），**腿部钟形感受器从未做过**。

## 六、产物与边界

- 脚本：`/run/media/sensen/Data2/cell_wound_prototype/run_campaniform_composition_audit.py`
- 测试：`/run/media/sensen/Data2/cell_wound_prototype/test_campaniform_composition_audit.py`（5项通过）
- 结果：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/campaniform_composition/campaniform_composition.json`
- 检索报告：`/run/media/sensen/Data2/cell_wound_prototype/RESEARCH_CONTACT_SENSING_IDENTITY.zh-CN.md`、`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/research_contact_identity_sources.json`

边界：单一注释release；外部标注取自检索账本，本审计未重新查询VFB；泛类标签CS/bCS不含节段信息；本审计未改动通道、增益或任何标签。
