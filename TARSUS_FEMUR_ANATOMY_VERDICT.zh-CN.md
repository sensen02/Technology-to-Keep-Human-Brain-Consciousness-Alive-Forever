# 跗节/股节解剖与扩展：最终结论（v4 已完成）

**手工研究原型，不是成品。** 这是这条线的收尾：解剖来源、我实测的数字、四个扩展变体的结果、以及明确的边界。

## 一、可用（有来源，可直接引用）

| 项目 | 来源 | 内容 |
|---|---|---|
| 跗节关节**真实轴向** | Haustein et al. 2024（Front Bioeng Biotechnol 12:1357598），开放数据`doi:10.7910/DVN/XWREDA` | 每腿给出**两髁点坐标**；hinge轴过两髁，故可**推导**（本轮的v4用的就是这个） |
| 跗节关节**范围/被动参数** | flybody（TuragaLab，Apache-2.0） | 轴`1 0 0`；T1 −0.7…1.2 / T2 −1…1.8 / T3 −0.8…1.2；逐腿springref；stiffness .01、damping .01 |
| 跗分节**耦合** | flybody | tarsus2–5由**一个**执行器经`<fixed>` tendon以**1:0.5:0.5:0.5**驱动 |
| **站立时该关节屈曲** | Azevedo 2024 Nature补充方法（开放获取） | "THE JOINT IS FLEXED WHEN THE FLY IS STANDING"；depressor=伸展、levator=屈曲 |
| 跗节内**无肌纤维** | FBbt；Oeftger 2026 iScience（开放获取） | 支持**只需一个**胫节-跗节自由度 |
| **跨关节长腱**作用 | Oeftger 2026 | retractor unguis自股/胫节肌经股胫关节腹侧至第5跗节，**被动稳定股节以远所有关节** |

## 二、我实测的数字

**真实轴是斜的，约定轴是垂直的**（坐标系无关判据）：

| 量 | 数值 |
|---|---|
| 数据集（μCT髁点）轴与胫节夹角 | 均值 **76.78°**（62.59–90.41） |
| 本项目约定轴与胫节夹角 | 均值 **90.35°**（89.92–91.45） |
| 差 | **13.56°** |

**v4按体节相对基完成映射**：把源数据的轴表示在"胫节方向+股节方向定义的平面法向+叉积"基里，再在**本模型对应几何**上重建同一组系数。映射后逐腿的"轴-胫节夹角"与来源**逐腿完全一致**（LF 77.35、LM 62.59、LH 90.41、RF 77.35、RM 62.59、RH 90.41），这说明**映射算术正确**（但不证明frame对应关系本身）。

## 三、四个变体：**同幅变差**

| 变体 | 轴 | 范围/被动 | 逃逸时间 | 200ms质心高 |
|---|---|---:|---:|---:|
| 25μs基线 | — | — | **23.15ms** | 0.9071mm |
| v1 声明几何 | 声明`0 1 0` | ±1.2 / 无 | 8.75ms | 0.5514mm |
| v2 FlyBody | 推导平行胫节 | FlyBody | 8.75ms | 0.5466mm |
| v3 +=springref | 同上 | +屈曲方向springref | 8.75ms | 0.5454mm |
| **v4 μCT解剖轴** | **体节相对基映射** | FlyBody | **8.72ms** | 0.5223mm |

**四版全部同幅变差**，因此：**轴、范围、被动刚度、弹簧静止位都不是原因**。

## 四、结论

1. 把胫节-跗节关节**松开**，在本模型上无论用哪种有来源的几何，都会**同幅加速崩塌**（约2.6倍）；
2. 剩余两个候选：**(a) 缺失的跨关节长腱被动稳定器**（动物有、本谱系没有）、**(b) 没有跗节感觉**；
3. 长腱的路径/长度/滑车几何**没有任何公开来源**，因此**不能用声明式几何硬加**——这正是v1–v3失败的方式；
4. FlyMimic把跗节**焊死是有原因的**：其X射线数据"ends ~¾ the length of the tibia… The tibia-tarsus joint is not visible"。

**建议：在这一谱系上停止加跗节自由度**；若要继续，只有两条路——整体切换到flybody（不同谱系、执行器集合不同、会打断BANC映射），或取得腱的几何来源。

## 五、已确认**不存在** vs **未解决**

**确认不存在**：跗节depressor/levator附着点坐标（成像截断）；实测ROM（来源自述约束是"empirically adjusted"）。独立佐证：FlyMimic的37个PathPoint只挂在5个frame上、**无任何跗节frame**；15块肌肉只在左前腿。

**未解决（非否定）**：SimTK是否存在果蝇模型（`www.simtk.org` 200，但搜索JS门控且真实端点在本环境被拒）；FlyMimic论文提到的中/后腿肌肉重建（7/8块）未发布。

## 六、本轮附带发现的结构不对称

**中腿与后腿没有Trochanter体**（只有LF/RF有）：六条腿的body集合并不一致。这本身是又一处"六腿复制"的裂缝，已记录在v4映射代码与结果里。

## 七、产物与验证

- v4构建：`/run/media/sensen/Data2/cell_wound_prototype/run_tarsus_dof_v4_anatomical.py`
- v4结果：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/tarsus_dof/tarsus_dof_results_v4.json`、`tarsus_dof_v4_axis.json`、`tarsus_dof_traces_v4.npz`
- 四版汇总：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/tarsus_dof/tarsus_dof_variants.json`
- 图：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/tarsus_dof/tarsus_dof_comparison.png`（已查看）
- 解剖轴审计：`/run/media/sensen/Data2/cell_wound_prototype/run_anatomical_joint_axis_audit.py` + 其测试
- 数据集落地：`/run/media/sensen/Data2/cell_wound_prototype/data/kinematic_leg_model/`
- 检索报告（20个来源）：`/run/media/sensen/Data2/cell_wound_prototype/RESEARCH_TARSUS_FEMUR_ANATOMY.zh-CN.md`
- 回归汇总：**117项测试通过，0失败**

边界：frame对应关系为**假设**（映射算术已自检）；数据集为单雌蝇运动学模型；"长腱是被动稳定器"是有来源支持的**假设**，本轮未单独证实；未使用任何未标注坐标。
