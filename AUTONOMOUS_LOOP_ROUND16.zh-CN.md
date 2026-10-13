# 第十六轮：站姿搜索无解 + 外部一手文献核对

目标未完成，飞行暂缓，手工仿真原型。本轮不含控制器、目标或动作表。

## 一、有界站姿搜索：没有找到可行站姿

在统一 coxa/trochanter pitch 族（含两档就座深度、15×15 网格、根部高度与 x/y 扰动）中共评分 **578 个姿态**：573 个仅跗节着地、0 个非足部位触地、5 个无接触、**可行数 0**（容差 L1<1e-6）。

最好姿态（仍不可行）：六腿统一 coxa-pitch +0.0357 rad、有效跗节穿透 0.100 mm、25 个接触、仅跗节、胸节 1.9969 mm。
- 全部90执行器 L1 = **3.7285**（约体重的15%）；摩擦锥 K=4/8/16/32 → 3.7607/3.7285/3.7126/3.7102，**已收敛**；
- 44 候选 L1 = 9.1133，明显更差；
- **受控比较**：相同有效穿透 0.100 mm 下，关键帧姿态重就座为 4.8249，最好搜索姿态 3.7285 确实更好，但都远不可行。关键帧原样为 5.0156。

## 二、一个影响解释的方法学发现（重要）

残差在浅就座时**由接触生成决定，而不是姿态**。关键帧姿态在不同有效穿透下的 L1：

| 穿透 mm | 0.000 | 0.005 | 0.010 | 0.020 | 0.040 | 0.0514 | 0.060 | 0.080 | 0.100 | 0.120 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L1 | 42.45 | 42.47 | 42.37 | 42.41 | 6.72 | 5.02 | 5.01 | 4.85 | 4.82 | 4.77 |

零穿透时 MuJoCo 只生成**1个**接触点——这解释了上一轮"恰好触地姿态 L1=42.45、看似结构性不可支撑"的假象。**比较姿态必须匹配有效穿透深度。**

父会话独立复核：从原样关键帧逐步下沉，接触数 14→16→18→23→27→29→33→39→43 平滑增加且始终仅跗节；原样关键帧 14 个接触、穿透 0.0514 mm。与该曲线一致。

**新记录的陷阱**：自由关节 `qpos[3:7]` 是**四元数不是欧拉角**；直接写入 0.03 会破坏四元数并把身体抬高约 3.5 mm，产生假的穿透值。脚本已改为四元数复合+归一化。`mj_mulM` 逐列装配质量矩阵的既有陷阱继续遵守。

## 三、外部一手文献核对（本轮下载并逐条核对）

详见 `/run/media/sensen/Data2/cell_wound_prototype/EXTERNAL_EVIDENCE_EON_VNC.zh-CN.md`。要点：

- **Eon Systems 官方说明自承**：身体"not currently driven by the full downstream motor hierarchy"；行走是"slight modifications to existing NeuroMechFly controllers, **trained to imitate** the walking behavior"；DN→身体映射"can be somewhat arbitrarily chosen by hand (**as is our case**)…still an approximation of the true motor hierarchy"；用 **DNa01/DNa02（转向）+ oDN1（前进）**——与本项目先前被否决的人工桥接同一思路。
- **独立评述**给出的 embodied 判据与本目标完全一致（脑→下行→VNC输入→VNC环路→运动神经元→关节力矩→运动），并指出 Eon 用动画控制器"decouples neural dynamics from motor output"。
- **"数字斯芬克斯"实验**证明行为逼真≠生物真实：线虫连接组+果蝇身体+强化学习解码器可产生"highly realistic fly walking—yet it is biologically meaningless"，且该连接组换成**随机RNN**同样可行。
- **MANC/VNC 2026 论文**：MN→肌肉对应在仅含VNC的EM数据里**无法直接观测**，只能靠树突形态比对文献；下行神经元**很少直接连**运动神经元；我们采用的四个前腿神经类（DProN/VProN/ProAN/ProLN）得到独立印证；生成行走节律的CPG神经元**尚未鉴定**。

## 四、综合判断

1. 塌陷不是"网络没学会"，也不只是"力气不够"：**该模型在该姿态族内无法静态支撑**（残差约体重15%，对锥加密稳健）。
2. 也因此，**继续在现有姿态上调神经参数没有意义**；真正的候选修复是有依据的物理/结构修正（腱路线/力臂、关节范围、粘附缺失）或逐腿非对称姿态——**均需外部证据**，不能由我们臆定。
3. 外部文献从另一侧支持同一条底线：**行为逼真不能替代生物保真**，人工接口（无论是手工阈值还是强化学习）都会让整个模型失去意义。

## 五、未探测的方向（如实记录，不是"已证明不可能"）

逐腿非对称姿态、胫节/跗节与 yaw/roll 姿态、修正腱路线/力臂或关节范围后残差是否消失、是否存在带粘附时的可行站姿。**本族搜索只覆盖六腿统一的 coxa/trochanter pitch。**

诊断 `/run/media/sensen/Data2/cell_wound_prototype/diagnostics/diag_posture_support_search.py`
数据与图 `/run/media/sensen/Data2/cell_wound_prototype/outputs/posture_search_round16/`（图已核看）
桌面 `/home/sensen/Desktop/cell_wound_prototype/autonomous_round16/`
