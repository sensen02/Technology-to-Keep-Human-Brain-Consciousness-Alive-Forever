# 小范围地形适应验收

人工搭建大脑项目身体侧 S2/S5。各子任务以同一冻结机构、原三球足端、完整 71 mesh 和 42 路有限 PD 做真实碰撞试验。原生积分步为 12.5 μs，控制决策 5 ms。保留自由根，脚只由关节驱动和地形触力移动。本次范围是身体台架，人工脑与真实颈部的因果验收按已登记的接口另行证明。

## 单个低凸台

静态凸台长 0.455、宽 0.70、高 0.032 原生长度。地面布局让右前、中、后足依次经过顶面，分别从约 2.875、15.633、28.457 秒开始出现顶面承重。地形感知教师采用基于实际身体与足端位置的几何规划，在 40.1 秒课程完成原生门，所有足垫越过远沿的时刻约 35.176 秒，各足在远侧实际重新承重，最晚左后足在约 35.616 秒。最终保持直立，原生滑移比约 0.374，实际前速约 0.10163 原生长度/s。非足载荷峰约 0.0591 W，处于工程 0.20 W 内，完整 D1 零非足触力资格分开保留。

延长到 43.3 秒的回合中，头胸腹最靠后的真实网格点从 39.280 秒起也越过远沿，停止时有 0.273 原生长度的余量。末窗完整 71 个网格的水平投影离开凸台实体占据范围，最小分离量 0.0282。原生滑移为 0.36311，实测前速为 0.10209，各足 7–9 个完整周期。两次各积分 3,464,000 步，独立重放逐字节相同。相同几何、没有凸台规划的纯步态教师也通过 40.1 秒工程门，滑移约 0.373。这一低台试验建立物理通行范围，尚没有建立抬腿反射对该低台全程的因果贡献。 [完整物理结果](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/block/EXTENDED_EVIDENCE.json) · [同几何配对](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/block/MATCHED_EVIDENCE.json)

单独把台高提高到 0.060 原生长度，配对检验绊脚抬腿。原生触力在 2.7247 秒出现，控制在 2.725 秒读到反向触力，2.730 秒起只改变右前腿动作。足端实际撤回约 0.089 原生长度，并在 3.068 秒起在顶面持续承重至少 0.02 W。开启反射的碰撞冲量为 0.0002432 W·s，关闭时为 0.0045356 W·s，降低约 94.6%。关闭组在 5.5 秒试验内未达到同一承重。该短试验复现的是局部真实触力工程反射的效果。归档的绊脚网络在特定新地形输入下出现提前触发，仍按旧覆盖范围管理。 [触力开关对照](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/block/TACTILE_EVIDENCE_V2.json)

## 浅缺口

利用已校验的真实局部缺口，在右前足预计落地但实际下探失载后，经足端雅可比及有限执行器收腿。右前足确实从原开口横向范围内直接移到远沿，整组足垫球全部过边后，在 3.22715 秒达到至少 0.02 W 的上沿连续承重。触发约 2.765 秒，远侧实测位置约 x=1.318、y=−0.866，缺口远沿为 x=1.272。直接复踏物理回放独立记录完全相等。相同位置填平时右前足约 2.788 秒开始承重，且无实际下探。冻结无反射对照较晚才通过右前足远沿。

保留了复踏后的失败。直接交回站立策略会把右前足拖回槽中。保持远沿足端锚点可以站稳，直接重新给推进指令仍可能在 50 ms 内把足拉回。修订后的 17.25 秒回合先在远沿实际承重约 0.458 秒，再抬足腾空返回站稳，重新给推进意图后安全走 9.6 秒并停稳。续走前速 0.07163 原生长度/s，原生滑移 0.48144，末端站稳，无非足载荷。两次各积分 1,380,000 步，独立重放逐字节相同。失载后的前段实际总足载始终至少约 0.995 W，至少五足承重；单列的主动换足补撑能力仍需与恢复和续走一同验收。 [直接远沿复踏原生审计](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/gap/DIRECT_FAR_AUDIT.json) · [安全交接与重放](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/gap/CLEAR_ACCEPTANCE.json)

继续使用原来的冻结低速续令跑到 53.25 秒，完成 4,260,000 步，末端站稳、原生滑移 0.49733，但续走后段停滞，整体有符号前速 0.02256 原生长度/s，未通过既有前进门。末端质心 x≈0.316，仍在缺口远沿 x=1.272 前。复查发现，旧示范检索在 20.845 秒锁在最后一行，此后推进意图持续有效，动作已接近零。该回合保留为需要持续步态的负例。 [连续长回合](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/gap/CLEAR_TRANSIT/NATIVE_AUDIT.json) · [动作检索诊断](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/gap/STALL_DIAGNOSIS.json)

改用连续世界足点的教师后，质心从 25.165 秒起越过缺口的远沿平面。两次仍在 34.128 秒同位置终止。左后足位于另一侧的 y≈1.026，接触法线被两块地板外露拼缝夹成相反的水平方向，之后出现质心下落与非足接触。按 y 方向重排地砖的替代方案保持原缺口尺寸，但右后足在 2.177 秒就卡到横向拼缝。两项地板实现负例均保留。 [连续足点试验](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/gap/ANCHOR_COURSE/RESULT.json) · [接触对审计](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/gap/X_SEAM_DIAGNOSIS.json) · [横向地砖试验](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/gap/Y_COURSE/RESULT.json)

保留右侧相同缺口，把另一侧脚道换成一整块静态水平地板后，使用同布局的关闭反射、填平对照重跑局部门。最早的右前足远缘承重仍在 3.22715 秒，其他五足稳住，实际接触与走前站姿门继续通过。 [同尺寸缺口新地板局部审计](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/gap/HYBRID_LOCAL_AUDIT.json)

修正地板后，41.25 秒全程通过。保留同一缺口恢复前段，6.450 秒撤掉制动并接连续足点规划。质心在 25.122 秒越过远沿 x=1.272，最后越过边界并重新持续承重的左后足在 36.764 秒通过。头、胸和腹部最靠后网格从 39.500 秒起保持过线，末窗该网格留有 0.02426 原生长度余量。停步后质心 x=2.85437，所有足垫的最小 x 至少为 1.73151，六足均实际承重。续走原生有符号前速 0.10614，滑移 0.27717，无非足载荷和额外力。两次各 3,300,000 原生步，姿态、动作、载荷和足端数据逐字节一致。缺口实际宽度只覆盖右前足脚道，其他脚在相邻地面走过同一个前后边界。 [长程原生门和足端逐个审计](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/gap/HYBRID_TRANSIT_ACCEPTANCE.json) · [头胸腹网格](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/gap/HYBRID_TRUNK_AUDIT.json) · [全程回放](/home/sensen/Desktop/engine修复验收/小范围地形适应-20261116T153927Z/浅缺口全程回放.mp4)

## 反射组合

同一身体侧入口根据当前接触和载荷选择标称动作、绊脚抬腿或踏空回收。原来通过的绊脚网络以冻结权重重新跑相同窄柱，对照只关闭该分支，动作前缀和原生轨迹相同；开启约提前 25 ms 复载，双臂均最终站稳。踏空分支是带限定作用时窗和实际接触反馈的工程教师；旧近侧恢复和新远侧复踏的连续回合均从同一入口逐字节复现。关闭近侧分支后出现相同实测失载，资格窗内没有重新获得上平面承重，末端没有稳定。对新凸台的独立触力抬足工程探针也通过同一选择器复现，沿用该探针自身的局部触力动作实现。本轮没有训练新的统一权重。 [共享入口](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/learn/compose.py) · [绊脚关闭对照](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/learn/PEG_ABLATION.json) · [踏空关闭对照](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/learn/NEAR_ABLATION.json) · [远侧复踏共享入口重放](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/learn/COMPOSE_FAR_CLEAR/ASSEMBLY_AUDIT.json) · [凸台共享入口重放](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/learn/COMPOSE_ENGINEERING_BLOCK/ASSEMBLY_AUDIT.json)

## 缓坡的原始试验

整片倾斜平面上的原平地控制器在 0.8° 和 2° 工况下实测下滑。平地锚点教师通过原生门，整体 2° 倾斜版本仍后退，工程门失败。有限三段盒的 2.3° 版本中，六足都曾实际在坡面承重，质心过坡顶并平均向前 0.08708 原生长度/s。坡前和坡顶两条接缝仍可能引入侧向接触，尾足停在坡前接缝附近，原生滑移 0.66910，非足载荷峰 0.80075 W，末窗未站稳。将三段盒压平的同口径对照也在 29.8 秒因持续不安全接触提前终止，因此三段盒结果不足以判定坡度本身是失稳原因。 [有限坡原始结果](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/learn/RAMP_PHYSICS/RESULT.json) · [0° 三段盒对照](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/learn/RAMP_MATCHED_FLAT/RESULT.json)

将地表整体替换为单高度场后，射线实测表面高度符合目标，2.3° 和 0° 对照均在同样的 0.225 秒终止。当时尚未迈步，两个工况的动作、姿态和原生力文件逐字节一致。替换地板已改变起始站姿的接触响应，应保留原平地接触才可继续比较有限坡。 [单高度场坡面](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/learn/RAMP_CONTINUOUS/RESULT.json) · [单高度场平地](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/learn/RAMP_CONTINUOUS_FLAT/RESULT.json)

下一版恢复原水平碰撞地板，将坡身埋在原地板下面，待 x=0.972 时从水平地面渐升，坡顶 x=1.572、高台升高 0.024098 原生长度。保存的起始位姿和原身体感官读数保持一致。25.7 秒验证中，左中、右前、右中足在坡面承担可测载荷，质心依次在 16.694、21.587 秒跨两端，净上升 0.012204。原生滑移 0.32858，原生前速 0.11387，非足载荷峰为零，撤令后站稳；末窗仍有尾足在坡前所以保留为局部通过。 [无接缝坡面短程结果](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/learn/RAMP_OVERLAY_ENTRY/RESULT.json)

相同 2.3° 结构的 41.7 秒长回合积分 3,336,000 步，五足在坡内实际承重，最终最靠后的足垫边缘 x=2.606，超过坡顶 x=1.572，站稳和全程原生行走门通过。前进 4.506 原生长度、原生前速 0.11280、滑移 0.30966，坡面非足门通过，末窗非足为零。走动结束时质心升高 0.006956 原生长度，完整验收仍欠预设的 0.012 抬升门。末态接触来源另按存档位姿和末动作直接积分 800 步测量，高台承担约 0.204 W 的竖直力，由两前足承担，原地面承担剩余约 0.796 W，由两中后足承担。全足通过纵向范围与全足在高台实际承重是两个测量结果。固定旧末态的局部试验可以抬高质心 0.0204，但未能停稳。配对原地板 0° 全程通过，滑移 0.26149。与较短坡回合前 21.306 秒、与平地回合前 2.356 秒的姿态和原生数据逐字节一致，分歧时刻单列。 [长坡结果](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/learn/RAMP_OVERLAY_LONG/RESULT.json) · [同初态配对审计](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/learn/RAMP_OVERLAY_AUDIT.json) · [分地形实际载荷](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/learn/PLATEAU_LOAD_SOURCES/RESULT.json) · [缓坡全程回放](/home/sensen/Desktop/engine修复验收/小范围地形适应-20261116T153927Z/缓坡全程回放.mp4)

## 飞行是否现在进入主线

我判断现阶段无需把飞行作为前置能力。地面已能构造局部触觉纠正和上层行为选择的独立对照。下一步适合固定通过的身体控制，在同一地面任务中检验头部感觉驱动的继续、停步或改道，推进和信息回传都经有来源的颈部通路。反射仍在身体就地闭环，可用不同通路的配对干预衡量上层决策变化。NeuroMechFly v2 已展示以地面运动做分层视觉、嗅觉与腿部反馈任务。 [NeuroMechFly v2 论文](https://www.nature.com/articles/s41592-024-02497-y)

当实验问题明确涉及起降、空中逃逸或只有飞行才能到达的资源，飞行有独立研究价值。可先做悬停、小转向和安全降落，在气动力和起降通过后验证走飞衔接。FlyBody 的飞行控制器和参考轨迹可用作未来教师来源。它训练自由飞行时把腿保持在收起姿态，因此项目仍需单独验证触地后的承重与行走衔接。 [FlyBody 论文](https://www.nature.com/articles/s41586-025-09029-4)

## 回放与运行记录

[低凸台全程](/home/sensen/Desktop/engine修复验收/小范围地形适应-20261116T153927Z/低凸台全程回放.mp4) · [凸台触力开关](/home/sensen/Desktop/engine修复验收/小范围地形适应-20261116T153927Z/凸台触力对照回放.mp4) · [浅缺口全程](/home/sensen/Desktop/engine修复验收/小范围地形适应-20261116T153927Z/浅缺口全程回放.mp4) · [浅缺口局部配对](/home/sensen/Desktop/engine修复验收/小范围地形适应-20261116T153927Z/浅缺口远沿复踏回放.mp4) · [缓坡全程](/home/sensen/Desktop/engine修复验收/小范围地形适应-20261116T153927Z/缓坡全程回放.mp4) · [命令与退出状态](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/RUN_CHECKLIST.zh-CN.md)
