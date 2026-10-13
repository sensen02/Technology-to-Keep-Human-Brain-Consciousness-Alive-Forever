# 跗节自由度扩展：闭环已跑，结论是明确阴性

**手工研究原型，不是成品。** 这是你授权的模型扩展的第一项完整实验：结构 + 真实运动池 + 三臂闭环。

## 一、做了什么

从**零下限分支XML**（而非规范XML，以保证ctrlrange等设置与已发布25μs基线一致）生成扩展模型，只新增、不改动既有joint/actuator/几何/质量/惯量：

| 维度 | 基线 | 扩展 |
|---|---:|---:|
| nq / nv / nu | 49 / 48 / 90 | **55 / 54 / 102** |
| 新增自由度 | — | **6**（每腿一个跗节铰链） |
| 新增执行器 | — | **12**（每腿 depressor + levator） |

**新执行器接的是真实注释运动池**（六腿双侧齐备）：

| 腿 | depressor | levator |
|---|---:|---:|
| LF / LM / LH | 3 / 4 / 5 | 1 / 1 / 1 |
| RF / RM / RH | 5 / 3 / 3 | 1 / 2 / 1 |

实测新执行器收到的命令最大 **0.779**（均值0.0093），说明它**确实被真实神经驱动**，不是空挂。

## 二、声明的假设（全部数值化，未调任何增益）

- **A1 关节**：插入在既有`Tarsus1`体内。v2中轴**由几何推导**（静止时与同腿胫节pitch轴在世界系平行，再换算到跗节本地系），**范围与被动stiffness/damping取自FlyBody**；v1为声明版本，仅用于对照。
- **A2 执行器**：参数整份复制既有胫节肌肉（gainprm/biasprm/dynprm一致），**不引入新调参**；ctrlrange沿用分支的`0 1`。
- **A3 附着点**：父侧`(0,0,0.10)`、子侧`(±0.02,0,0)`，**声明杠杆臂≈0.020**；本项目无这些肌肉的附着点测量。
- 感觉前端**未改**：新跗节关节不进入关节偏差组，传感映射仍用原来7个关节。

## 三、结果：**两个独立几何参数化都是明显更差**

| 指标 | 25μs基线 | v1 声明轴/范围 | v2 **FlyBody来源**轴/范围+被动阻尼 |
|---|---:|---:|---:|
| 10%胸高逃逸时间 | 23.15ms | **8.75ms** | **8.75ms** |
| 200ms根高 | 0.8315mm | 0.8454mm | 0.8356mm |
| 200ms质心高 | 0.9071mm | **0.5514mm** | **0.5466mm** |
| 200ms非足接触 | — | 18 | 22 |

v2 的改动（全部有来源）：轴**由几何推导**使新铰链在静止时与**该腿已有的胫节屈曲轴平行**（对应FlyBody"胫节与跗节同一屈曲轴族"的约定；推导结果为本地系y，与v1的声明值一致，属交叉验证）；范围与被动参数取自FlyBody（逐腿范围T1 −0.7…1.2 / T2 −1…1.8 / T3 −0.8…1.2；`leg`类 stiffness 0.01、damping 0.01）。

**两版结果几乎相同**，说明**变差不是我编的范围/轴造成的**，而是"在缺乏跗节感觉与真实附着点的情况下加一个自由跗节自由度"本身造成的。

**加了自由度后，逃逸时间缩短到基线的约38%。** 图已实际查看：质心曲线在约25ms内就掉到0.6mm并继续下探，而基线同期还在1.5mm附近。

对照全部通过：命令回放8类数组**精确一致**、输出静默时`act`/`ctrl`严格为0、全部状态有限、源码哈希未变、12个新执行器都有池。

## 四、这说明什么（如实）

- 这与我在实验前写的预期一致：**在没有任何跗节附着点测量、也没有跗节感觉的前提下，凭空加一个自由自由度会引入新的塌陷模式**，而不是提升站立。
- 它**不证明**跗节在动物身上不重要；它证明的是**按当前可得的证据去加这个自由度，代价大于收益**。
- 因此"把行走所需的自由度补齐"这条路，**不能靠声明式几何硬塞**；要么取得跗节肌肉附着点与关节轴的实测/解剖来源，要么接受当前模型的能力边界。

## 五、产物与验证

- 结构构建：`/run/media/sensen/Data2/cell_wound_prototype/run_tarsus_dof_structure.py`、`test_tarsus_dof_structure.py`
- 池导出（需pandas解释器）：`/run/media/sensen/Data2/cell_wound_prototype/tools_export_tarsus_motor_pools.py`
- 闭环实验：`/run/media/sensen/Data2/cell_wound_prototype/run_tarsus_dof_branch.py`、`test_tarsus_dof_branch.py`（5项通过）
- 结果：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/tarsus_dof/tarsus_dof_results.json`
- 轨迹：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/tarsus_dof/tarsus_dof_traces.npz`
- 图：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/tarsus_dof/tarsus_dof_comparison.png`
- 扩展XML：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/tarsus_dof/fruitfly_tarsus_dof.xml`
- 回归汇总：**108项测试通过，0失败**

边界：单一姿态、单一timestep、单一tier；新增关节的轴与附着点是**声明假设**而非测量；感觉前端未包含跗节；结果更差属有效结果，不得当作治愈或行走证据。
