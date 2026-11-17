# 身体侧连续调速与双向出弯

人工搭建大脑项目 S2/S5 的身体台架试验。按用户指定由三位子代理分别做连续调速、双向出弯、监督候选与组合复验。全身沿用冻结 71 mesh、原三球足、42 路有限 PD、无附着、12.5 μs 原生步、5 ms 决策和同一名义站姿。台架下发的两个连续意图是推进速度（原生长度/s）和有方向的偏航速度（rad/s）。脑侧及真实颈部的途径按独立接口验收。

## 连续调速的同身教师

| 实测正向均速（原生长度/s） | 原生最大滑移比 | 同时验证六足换步和两端站稳 |
| ---: | ---: | --- |
| 0.071713 | 0.481386 | 通过 |
| 0.077382 | 0.477320 | 通过 |
| 0.083425 | 0.490472 | 通过 |

教师使用连续推进意图调节原冻结步态参考中的进度，实际动作只经过有限关节执行器。在同一回合用平滑的上升和下降指令，实测各窗的有符号前速约为 0.072021、0.076368、0.084002、0.076249、0.070847。整体原生滑移比为 0.477143，完成六足完整周期与撤令停稳。各附录短窗的滑移比约 0.405 至 0.619，短于完整步态周期时，分母和承重阶段选取会影响比值。这些短窗不计作逐时通过。刻意更慢的档位未通过原 0.50 门。相对于原型，最慢两档的最大绝对足端滑速降低，但身体前速降低得更多。 [常速资格与旧动作回归](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T075254Z/speed/BASE_PACE_QUALIFICATION.json)

另一条工程实现从当前执行器目标估计物理步态位置，用当前推进意图选下一步动作。它在相同斜坡下通过原生门，平均前速 0.077331，滑移比 0.477145。2,260 条决策分别按正序、倒序和随机顺序输入，逐条复现；独立 904,000 步回放也完全相同。 [当前观测调速复验](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T075254Z/speed/OBSERVED_QUALIFICATION.json)

## 左右弧线与出弯的同身教师

教师沿用原直走进入左右弧线，并用同一套足端接触锚点延续动作，在 1.2 秒内连续减少偏航。渐消结束后沿实际出弯方向继续走直线，并做有限的航向修正。左右弧线净转角约 +56.77° 和 −57.35°。两侧出弯、直走与停稳均达到分支的原生工程门，完整六足都有恢复周期。撤销偏航之后的短时航向修正与后续纵向走动分别保存。

| 方向 | 旧试验直接切回直走 | 同前缀直接切换，本轮对照 | 渐消偏航后滑移，4.8 秒旧窗 | 渐消加后续完整 6.4 秒滑移 |
| --- | ---: | ---: | ---: | ---: |
| 左 | 2.112 | 1.778 | 0.433 | 0.464 |
| 右 | 1.560 | 6.745 | 0.300 | 0.358 |

新匹配直接切换的右侧实际前进约 0.0062 原生长度/s，较低的推进分母显著影响其滑移比。保留绝对足滑速度与前进证据。两个新教师保留转向指令渐消和随后沿自身实际朝向走路的全过程。 [左转原生结果](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T075254Z/turn/TEACH_LEFT_A/RESULT.json) · [右转原生结果](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T075254Z/turn/TEACH_RIGHT_A/RESULT.json)

上述左右两条教师各独立重积分 1,800,000 步，原生碰撞、动作和标准轨迹逐字节一致。另一组采用更缓的航向微调，在随后 5.3 秒零偏航命令段的原生滑移比分别是 0.3301、0.3100，实际净偏航 +0.150°、−0.201°。六足在两组完整 8.0 秒出弯窗均完成换步，末端站立也通过。 [更缓航向调节结果](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T075254Z/turn/SOFT_TRIM_EVIDENCE.json)

## 用同身轨迹监督候选

首个候选用通过的动作数据拟合当前 213 项本地状态和两个实时意图到 42 路有限执行器。保留原型动作并学习有界修正。数据仅使用已验收同身物理轨迹。首批共 15,998 行，训练 240 次更新。拟合的单样本剔除误差约 0.01259，应用残差后对应训练误差约 0.01159。最终行为以保存后物理闭环与独立复验判断。 [首个训练检查点登记](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T075254Z/learn/FIT_COMBINED_INITIAL/FIT_RESULT.json)

首轮保存候选在直线升降速回放中实现 0.072108 → 0.084077 → 0.070810 原生长度/s，整体原生滑移比 0.477479，并通过六足周期、支撑和起止站立。在与左转教师匹配的进入条件下，候选实际转角 +56.74°，4.8 秒出弯原生滑移比约 0.432939，同次继续走直线并停稳。首轮右转实际转角仅 −0.49°，未过转向与直行门。观察动作检索发现，近零偏航误检索对侧转弯示范，随后部分区间还退回直走示范。初批合成回放在变速后换步相位变化时也未产生预期左转，完整合成结果待按补充试验区分。 [学习调速原生审计](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T075254Z/learn/STUDENT_RAMP/NATIVE_AUDIT.json) · [学习左转与出弯](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T075254Z/learn/STUDENT_LEFT/RESULT.json)

给同一检查点的样本检索增加偏航符号约束后，右弧恢复到 −57.349°，旧 4.8 秒出弯窗滑移 0.299660。相同约束的左弧重放仍是 +56.740° 和 0.432939，标准轨迹和原生数组完整匹配约束前的左弧成功回合。右弧的独立稳定性对照也逐字节匹配，同一权重在另一组未入库的加减速设定中通过，滑移 0.477312。步态动作网络每拍只接收当前局部状态与当前两维命令。左右弧线的台架命令包含按实测航向生成的短时微调，属于台架输入的一部分。 [左弧方向约束后](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T075254Z/learn/LEFT_SIGNED/RESULT.json) · [右弧方向约束后](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T075254Z/learn/RIGHT_SIGNED/RESULT.json) · [留出调速](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T075254Z/learn/RAMP_HELD_SIGNED/NATIVE_AUDIT.json)

## 组合回合

未对齐入弯相位的学习回放先组未形成有效左转。方向约束后，同一课程能转到 +50.79° 和两端站稳，但零偏航段沿所得朝向的实际前速为 −0.01710 原生长度/s，滑移比 1.392，出弯失败。初版工程组合也是用真实有限执行器跑完，但弧段及完整出弯滑移超限，保留两条失败证据。

在连续加减速后延长约 4.03 秒低速直走，使入弯回到已覆盖的步态相位，并使用较缓的航向微调。此工程教师真实完成单回合的 **32.93 秒**站立、上升和下降调速、左弧转向、渐消偏航、零偏航直走及撤令停稳，共 2,634,400 个原生步。左转 +56.812°，弧段原生滑移 0.419600，旧 4.8 秒出弯窗滑移 0.398290，完整出弯窗滑移 0.383573，零偏航直走窗滑移 0.320567。这组结果适用于给定课程和同一初态。该回合生成一条补充物理示范，用于下一版学习候选。 [工程组合物理结果](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T075254Z/learn/INTEGRATED_PHASE_LEFT/RESULT.json)

第二版加入组合教师后用 **22,243** 行真实决策训练 240 次更新。保存后的同一份权重独立加载，在真实台架完整跑过该左转课程。学习策略接收工程教师已记录的两列高层指令及自身当前身体观测，以自己的动作重新进行整个物理积分。入弯的命令切换有一个 5 ms 零推进样本，物理轨迹连续。切换后首个 100 ms 实测平均位移速度约 0.051 原生长度/s，始终以物理回放审计。工程教师本身做了 2,634,400 步独立重复，完整的物理轨迹、动作和原生接触记录逐字节匹配。

| 第二版的完整左转组合回合 | 实测 |
| --- | ---: |
| 调速低档 → 高档 → 降速后，有符号前速 | 0.072108 → 0.084076 → 0.070812 原生长度/s |
| 弧段实际净转角 | +56.778° |
| 弧段原生滑移比 | 0.418903 |
| 旧 4.8 秒出弯窗口原生滑移比 | 0.397972 |
| 完整 8.0 秒出弯原生滑移比 | 0.383288 |
| 零偏航直行，有符号速度／实际漂移 | 0.087699 原生长度/s／+0.123° |
| 零偏航直行原生滑移比 | 0.335248 |
| 六足换步，撤令后站稳 | 通过 |

这份权重也在匹配的独立右弧课程达到约 −57.349°，4.8 秒出弯原生滑移比 0.299665、完整出弯 0.357958，并停稳。整段左转的 6,586 个动作与独立右弧的 4,500 个动作按倒序独立重算逐位一致。出弯输入经过方向检索约束，台架使用明示的航向微调，完整左转采用相位对齐和指定命令。当前完整组合实证范围为左转，独立右弧接直走也已验证。完整右向组合、不同初态与未见的入弯相位待复验。 [第二版课程原生验收](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T075254Z/learn/LEARN_PHASE_LEGACY_AUDIT.json) · [同权重右弧](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T075254Z/learn/RIGHT_COMPOSITE_WEIGHT/RESULT.json) · [当前观测动作复验](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T075254Z/learn/CURRENT_ONLY_COMPOSITE.json)

## 交付

- [学习策略完整组合回放](/home/sensen/Desktop/engine修复验收/连续调速与转向衔接-20261116T075254Z/学习策略完整组合回放.mp4) · [同次关键帧](/home/sensen/Desktop/engine修复验收/连续调速与转向衔接-20261116T075254Z/学习策略完整组合关键帧.png)
- [加减速实测曲线](/home/sensen/Desktop/engine修复验收/连续调速与转向衔接-20261116T075254Z/加减速实测曲线.png) · [学习策略右弧回放](/home/sensen/Desktop/engine修复验收/连续调速与转向衔接-20261116T075254Z/学习策略右弧接直走.mp4)
- [调速运行与失败](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T075254Z/speed/REPORT.zh-CN.md) · [转弯运行与退出](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T075254Z/turn/RUN_CHECKLIST.zh-CN.md) · [监督训练及物理复验命令](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T075254Z/learn/RUN_CHECKLIST.zh-CN.md)
