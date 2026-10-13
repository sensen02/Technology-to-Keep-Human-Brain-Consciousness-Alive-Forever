# 跗节/股节解剖来源：已找到一份**有来源的关节定义**

**手工研究原型，不是成品。** 只读提取，未改动任何模型，未跑任何行为实验。

## 一、上游模型追溯（本轮新查）

- 本地`outputs/upstream_muscle_round11/README.md`确认：规范模型源自 **FlyMimic**（`github.com/gizemozd/FlyMimic`，Apache-2.0），论文为 *Musculoskeletal simulation of limb movement biomechanics in Drosophila melanogaster*（ICLR 2026，Ozdur/Ramdya 等），原始肌肉模型开发指向 `neuromechfly-muscles`（该仓库已404）。
- 我核对了上游仓库：其 `assets/models/opensim/` 只有**与我们本地完全相同**的两个`.osim`（483,413 / 491,132字节，逐字节同大小），**没有更多关节或肌肉**。
- 结论：**跗节/股节自由度不是本项目转换时丢掉的，上游模型本身就没有。**

## 二、找到的来源：FlyBody（已随flygym安装）

已安装的`flygym`包内含 **FlyBody** 模型：`assets/model/flybody/fruitfly.xml`。它**有跗节关节**：

| 关节 | 轴（该模型body本地系） | 范围（rad） |
|---|---|---|
| tarsus（T1/T2/T3） | `(1 0 0)` | −0.7…1.2 / −1…1.8 / −0.8…1.2 |
| femur extend | `(1 0 0)` | −0.15…2（T1）/ −0.7…1.5（T3） |
| femur twist | `(0 1 0)` | −1…1 |
| tibia | `(1 0 0)` | −1.35…1.3 |

每条腿**8个执行器**（coxa 3 + femur 2 + tibia 1 + tarsus 2），即**跗节与股节都被建模**。

## 三、这允许什么 / 不允许什么

**允许（可引用为来源）**
- 跗节-胫节关节（以及更多跗分节）在已发表果蝇模型中**确实存在**；
- 跗节关节的**范围与弹簧参考**可直接引用，不必再"声明发明"；
- 腿部屈曲轴族在胫节与跗节上**是同一个本地轴**；
- 股节用 extend + twist 两个轴建模。

**不允许**
- **不能直接照抄轴数值**：两个模型的body朝向不同，轴必须**通过节段几何对应**来换算，而不是按数值搬；
- **没有任何肌肉附着点**：FlyBody驱动的是**关节**而非单块肌肉，因此它**不提供**跗节/股节肌肉的起点与止点；
- 不得据此推断递质、连接或运动池。

## 四、因此本项目的下一步（尚未做）

1. 用FlyBody的**跗节范围**替换我上轮声明的±1.2 rad，并把轴**通过节段几何**换算到我们模型的tarsus本地系（不是按数值搬）。
2. **肌肉附着点仍是空缺**：FlyBody无法提供，只能等解剖学来源（检索任务进行中）或继续声明为假设。
3. 若重跑扩展，仍必须带**正常/静默/回放**三臂对照，并如实报告——上轮"声明式几何"版本把逃逸时间从23.15ms压到8.75ms，属明显变差。

## 五、产物与验证

- 提取脚本：`/run/media/sensen/Data2/cell_wound_prototype/run_flybody_anatomy_audit.py`
- 测试：`/run/media/sensen/Data2/cell_wound_prototype/test_flybody_anatomy_audit.py`（4项通过）
- 结果：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/flybody_anatomy/flybody_anatomy.json`
- 上游文档（下载）：`/run/media/sensen/Data2/cell_wound_prototype/data/upstream_flymimic/`（含转换验证PDF，已`pdftotext`读取：只讲力矩臂与力-长度验证，**不含**附着点解剖）
- 回归汇总：**112项测试通过，0失败**

边界：单一已安装版本的flygym资产；两模型body坐标系的对应关系目前只按约定断言、未证明；本审计未修改模型、未跑episode。
