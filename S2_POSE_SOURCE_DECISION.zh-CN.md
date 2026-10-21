# 初态来源决策：停止将搜索派生姿态当生物站姿

手动只读研究原型，非成品。没有新模型加载/forward/step/姿态搜索。对S2/A门来源与下一任务作明确决策，不变原严格目标。

## 已追清的构造链

1. canonical XML本身是tools_stand_and_ground.py生成的standing variant：六腿声明neutral、root高度/各腿coxa offsets使脚触地；头注释还记录40附着点和75force-length bands工程修订。keyframe是模型参数，不是已提供的动物测量。
2. outputs/stance_margin_round37/stance_margin.json记录6000次搜索，status=STANCE_SUPPORT_MARGIN_SEARCH；tools_stance_margin_search.py按肌肉平衡资格优先，再优先较大COM margin，固定随机种子/随机起点/逐坐标搜索。它不是动物站姿crosswalk。
3. stance_for_reflex.json的42偏移与search.best.x精确一致。run_fix_round39.apply_round37_pose将偏移加到编译keyframe，然后prepare_body重新seat_root；JSON root_z未在apply中读取。旧docstring cm说法已由canonical单位更正，不能复活。
4. 固定刺激协议再将5个越界标量关节clip到原范围，然后重新就座。此显式派生是工程诊断选择，不自动得到支撑/生物资格。

离线使用已保存编译地址，重建key+offset与historical_qpos除z外精确相同；再按保存jnt_range投影与projected_seated_qpos除z外精确相同。root z：key2.04535、historical2.044880332462318、projected2.0486760924304157模型长度；本轮没有再次运行seat_root验证高度计算，来源代码与归档数值分开。5变更分别LF coxa_pitch/LF trochanter_pitch/LM coxa_roll/RH coxa_pitch/RM coxa_yaw，原值与投影值见桌面chain.json。

这也解释为何不能把旧搜索best.feasible=true直接当当前证书矛盾：搜索时make_qpos会对key+offset先clip，当前历史apply曾未clip、下游才投影；模型/版本/力域细节也不同，旧字段不是当前严格数值证明。本轮不完整复现旧搜索。

## 重要标签更正

首次离线核对误把XML文本joint顺序当编译顺序，导致断言退出1；检查既有metadata后发现加载器重排。改用保存dof_labels/jnt_qposadr完成重建退出0。此前证书报告中的“主要DOF名称”受同样错误影响：

|DOF|更正后的编译名|
|---|---|
|25|joint_LMTrochanter_roll|
|23|joint_LMTrochanter_yaw|
|16|joint_LHTrochanter_yaw|
|18|joint_LHTrochanter_roll|

之前写的25 LMCoxa_pitch、23 LMTrochanter_roll、16 LHTrochanter_roll、18 LHCoxa_pitch均撤销。数值A/y/Fraction证书未使用关节名，所以不可行结论不受标签错误影响。旧归档不改写，新增更正必须与其一起阅读；权重不是单关节故障原因。

## 现在选择什么，而不是继续泛审计

- 不直接换XML keyframe来声称生物站姿；它只可作为明确工程初态，且尚未在本轮取得支撑资格。
- 不自动从6000次搜索历史中挑一个能过LP的姿态，避免表现选择。
- 站立门暂停当前初态路线；若要重新启动，需要来源明确的初态/身体修订协议。当前只核上述限定来源链，不能声称整个本地所有数据集不存在动物姿态。
- **下一可立即推进项选择B门的末端因果诊断**：先用已有映射的一条腿/少数池，声明身体可不承重，明确脉冲→池输出→活化→局部力矩/短时位移；正常、运动切断、命令重放/被动对照，单列初态假设。不能把非承重或外部约束下局部动作当站立。先查已有P1/固定刺激是否已覆盖相同问题，再写只新增缺失观测的有界协议，涉及新支架/锁定/身体改版须确认。
- 与B门并行的站立准备是提供一个可选方案说明：有测量来源的姿态映射，或单列工程身体/初态诊断分支。未获确认不执行新姿态搜索/参数改版，也不发送维护者请求。

## 交付与限度

`/home/sensen/Desktop/engine修复验收/S2-pose-source-20261014-v1/chain.json`保存11路径SHA、6000搜索记录来源、5关节投影与编译名更正。XML/JSON/NPZ读与断言完成，未运行项目模块或模型/原生仿真/全量测试/网络/新图。初错误退出1后修正0，git diff --check另核；非独立审查。站立行走、生物标定和整体收敛仍未达成。

## v1.3 结案：关键点拟合不可辨识，不能作为站姿来源（2026-10-15）

**负面结论正式归档（不再只留在核验 JSON 里）**：

1. **拟合全程 range-bound**：42 个中位关节值（7 自由度 × 6 腿）**全部贴在关节限位上**；**21/42 在全部 17 帧被钉住** ⇒ 结果是**边界解**，不是内点最优。
2. **残差依赖未记录的旋转约定**：三点“点心化 Kabsch＋零行”是脚本特有约定；“apex→胸节原点”**不能固定绕翼轴的平面内旋转**。换约定后 **L1 0.8654→0.9201、R1 0.8489→0.7970**（前腿约 6%），并**有一条中位关节翻到对侧限位**。跨约定的比较**没有意义**。
3. **就座位姿被下游 gate 拒绝**：`FAIL_REJECTED_BY_GATE_NONFOOT_CONTACT` —— 41 个主动接触中 **40 个非足**（`floor|A3_geom`、`A4_geom`、`A5_geom`、`Haustellum_geom`、`LFCoxa_geom`…），thorax 由规范 **2.0487 降到 0.5407**（腹面塌地）；raw 姿态则 **ncon=0**（表面间隙 **1.45802**，geom 原点度量 1.50734）。
4. **结论**：**关键点拟合不能作为站姿来源；这条线到此关闭。**

## A 门状态：blocked-by-input（输入缺失冻结）

- **四条替代来源全部不可用**：①**XWREDA** 的初始姿态是**模型自身的合成正则化位形**（论文 Methods 2.6：调 yaw 让腿伸直、运动链里角度为 0）；②**Haustein 2024 没有站姿角度表、也没有补充材料**；③**BBNPYX 的 7 自由度映射 FAILED**；④**关键点拟合不可辨识**（见上）。
- ⇒ **“用搜索/拟合造一个可支撑姿态”这条路没有可用输入**；**规范声明姿态的静态支撑系统已证不可行**（精确 Fraction 分离证书，**gap = 0.002771012308801305**）。
- **禁止**把 **6000 次姿态搜索**或**任何 clip 就座**的产物当作站姿证据（它们只是工程诊断初态）。
- 姿态一律按**声明的模型参数**标注；**不再新增姿态搜索**；A 门记为 **blocked-by-input**。
