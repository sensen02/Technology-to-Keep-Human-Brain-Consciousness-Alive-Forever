# 平地左右弧出弯接直走

连续物理教师已在原机体上完成左右两条出弯衔接，包含静立、4.8 秒冻结直走入弯、9.6 秒弧行、6.4 秒连续出弯接直走和 1.2 秒撤令站稳。每条正式回合 22.5 秒、1,800,000 原生步。

方法是在同一个私有逆运动学几何参考中保留每条腿的摆动顺序和支撑期世界足锚，原弧线结束后在 1.2 秒内连续降低目标偏航，随后用实测航向重新对齐。行走目标逐渐减速，最后保持已建立的脚端支撑。控制全程走原有限 42 PD 关节接口。真实身体没有外力、附着、回合中重置或根轨迹写入。

| 原生量 | 左弧 | 右弧 |
| --- | ---: | ---: |
| 旧档案直接切换的首 4.8 秒滑移比 | 2.112 | 1.560 |
| 本轮同弧线前缀直接切换的首 4.8 秒滑移比 | 1.778 | 6.745 |
| 连续出弯首 4.8 秒滑移比 | **0.4333** | **0.2997** |
| 全 6.4 秒出弯滑移比 | 0.4637 | 0.3580 |
| 出弯首 4.8 秒沿阶段开始时体轴的前速（原生长度/s） | 0.09167 | 0.09207 |
| 弧线净转角 | +56.77° | −57.35° |
| 航向短校正后的直走净偏航 | +0.167° | −0.292° |
| 出弯完整换步 LF、LH、LM、RF、RH、RM | 1、2、1、1、1、2 | 1、2、1、1、1、2 |
| 末段站稳 | 通过 | 通过 |

新直接切换对照与连续组在 14.900 秒前逐位一致。其右侧沿换向时体轴前速只有 0.00620 原生长度/s，归一化滑移分母变小。实际最坏单足载荷加权切向速度也由 0.07753 降至 0.02759。左侧同值由 0.11950 降至 0.03974。旧档案及新同前缀对照是两套独立参照，各自保持原口径。同前缀对照证明直接切换的问题在本轮匹配的实际来态中仍会出现。

两方向首条合格示范已通过散列及原生准入，供学生拟合。每行保存 213 路当前身体传感、单位明确的两路意图和实际有限 42 路动作。教师的足锚、看向落脚点的规划和相位时间留在教师算法内。首条合格左、右示范在 1.2 秒偏航渐消后，都还有 0.45 秒的短航向校正，实际分别反向约 3.71° 和 3.10°。它证实纠正后的持续直行，完整实际偏航没有严格单调。这些证据独立于后续学习策略是否复现成功。

[左弧教师准入](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T075254Z/turn/TEACH_LEFT_A/TEACHER_ACCEPTANCE.json) · [右弧教师准入](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T075254Z/turn/TEACH_RIGHT_A/TEACHER_ACCEPTANCE.json) · [原生朝向与接触复核](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T075254Z/turn/PROFILE_EVIDENCE.json) · [实际两侧配对影片](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T075254Z/turn/VIDEO_CONTINUOUS/弧线出弯到直走配对.mp4)

两条首版示范分别在独立进程重积分全部 1,800,000 原生步。重放在每个 5 ms 的决策点逐位核对当前观测与工程命令，再执行原记录的有限动作。每方向三份 NPZ 原生力、毫秒轨迹和冻结决策的完整 SHA-256 与原试验逐位相同。

[左弧同前缀对照及重放](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T075254Z/turn/MATCHED_LEFT_WITH_REPLAY.json) · [右弧同前缀对照及重放](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T075254Z/turn/MATCHED_RIGHT_WITH_REPLAY.json)

### 较缓的实测航向校正

把航向校正从 0.45 秒展宽为 1.5 秒，并把连续出弯扩展为 8.0 秒。两个 24.1 秒完整回合也通过。补偿角速度峰值降到左 .0740、右 .0608 rad/s，均小于主弧线的 .115 rad/s。修正结束之后，输入偏航意图连续精确为零 5.3 秒，直到停车窗开始。该子窗实际原生滑移比左 .3301、右 .3100，前速 .08328／.08340，净偏航 +.150°／−.201°。全 8.0 秒出弯滑移比 .3982／.3189，首 4.8 秒为 .4171／.2959。六条腿在全出弯窗均完成周期。5.3 秒子窗中 LM 没有两个完整相邻的起落锚点，单独的窗口周期数保留原生记录。最终两个方向均再次站稳。两版共四条方向示范作为工程示范分别封存，独立逐位重新积分证据对应首版 A 左右两条。

[慢校正 / 真零偏航窗口原生证据](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T075254Z/turn/SOFT_TRIM_EVIDENCE.json) · [较缓航向校正录像](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T075254Z/turn/VIDEO_SOFT/弧线出弯到直走配对.mp4)
