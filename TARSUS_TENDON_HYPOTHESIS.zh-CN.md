# 跗节扩展：三版几何全部同幅变差，指向"跨关节长腱"

**手工研究原型，不是成品。** 本文档汇总三个**有标注**的扩展变体，并给出由已发表来源支持的原因判断。

## 一、模型谱系已确认：FlyMimic/Miller，**不是** flybody

本地核验：规范XML含`Miller28..33`标记、mesh `scale="1000"`、`angle="radian"`、`gravity 9801`——即FlyMimic（OpenSim→MuJoCo）谱系。子代理独立核验：FlyMimic的跗节关节在OpenSim里是**WeldJoint**，转换后**没有跗节关节**；而flybody有跗节+股节关节、78个执行器。**两者不是同一模型。**

**关键事实（Azevedo et al. 2024 Nature补充方法，逐字）**：其所用的X射线全息纳米断层数据
> "Distally, the dataset ends ~¾ the length of the tibia. … The tibia-tarsus joint is not visible."

即**胫节-跗节关节本来就没被成像到**——所以FlyMimic把跗节焊死是**有原因的**，不是遗漏。

## 二、三个变体的结果：**完全相同地变差**

| 变体 | 关节轴 | 范围 | 被动属性 | 逃逸时间 | 200ms质心高 |
|---|---|---|---:|---:|---:|
| 25μs基线 | — | — | — | **23.15ms** | 0.9071mm |
| v1 声明几何 | 声明`0 1 0` | 声明±1.2 | 无 | **8.75ms** | 0.5514mm |
| v2 FlyBody来源 | **推导**（与该腿胫节屈曲轴平行） | FlyBody逐腿 | FlyBody stiffness 0.01 / damping 0.01 | **8.75ms** | 0.5466mm |
| v3 += springref | 同上 | 同上 | 再加springref，**放在实测的屈曲方向**（来源明确：站立时该关节屈曲） | **8.75ms** | 0.5454mm |

三臂对照全部通过（回放精确、静默零驱动、有限、源码未变）。

**结论：变差与轴的取值、范围、被动刚度、弹簧静止位都无关。**

## 三、来源支持的真正原因

Oeftger et al. 2026（iScience，开放获取）逐字：长腿腱（=retractor unguis apodeme）
> "is controlled by long tendon muscles in the femur … and tibia … and runs ventral through the femoro-tibial joint and up to the end of the fifth tarsal segment where it connects to the unguitractor plate. It directs and helps to stabilize all joints distal to the femur, including the tarsal segments that lack muscles."

同一来源还指出**跗分节之间角度变化可忽略**、FBbt记录**跗节内没有肌纤维**——即**只需一个胫节-跗节自由度**（我做的正是这一个，方向是对的）。

**因此：焊接跗节是源模型对"跨关节长腱被动稳定"的替代；把关节松开，就去掉了稳定器，而没有来源给出这条腱的路径、长度与滑车几何。**

## 四、下一步的选项（都不含臆造几何）

1. **整体改用已发表的flybody设计**（跗节关节 + tarsus2–5的`1:0.5:0.5:0.5`固定腱耦合 + 逐腿springref）——但那是**另一个谱系**：执行器集合不同（78 vs 90），会打断BANC运动神经元→肌肉的映射；
2. 保留FlyMimic，**用单个等效执行器驱动跗节**，其杠杆臂**显式声明为假设**——这是上轮已试的路径，三版都变差，除非同时补上被动稳定；
3. **明确声明**当前模型无法忠实表达胫节-跗节关节，停止在这一支上继续加几何。

我倾向第3条，除非愿意承担第1条的谱系切换代价——因为**没有任何公开来源给出该关节的肌肉附着点与腱的路径**（这已被确认为"确实不存在"，而非"检索失败"：Azevedo补充方法说明成像范围止于胫节3/4处）。

## 五、产物

- 变体汇总：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/tarsus_dof/tarsus_dof_variants.json`
- 三版结果：`.../tarsus_dof_results.json`（v3）、`tarsus_dof_results_v1_declared.json`、`tarsus_dof_results_v2_flybody_nospringref.json`
- 检索报告（子代理）：`/run/media/sensen/Data2/cell_wound_prototype/RESEARCH_TARSUS_FEMUR_ANATOMY.zh-CN.md`、`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/research_tarsus_femur_sources.json`
- 回归汇总：**112项测试通过，0失败**

边界：单一姿态、单一timestep、单一tier；"焊接=被动稳定替代物"是**由已发表解剖事实推出的假设**，本轮实验**未能单独证实它**（三版都缺该腱，因此差异无法归因到单一因子）；未验证任何附着坐标。
