# 右前足浅缺口远侧复踏和续行

人工搭建大脑项目 S2/S5 身体台架。沿用已验收站姿、71 mesh 身体、三球足垫、42 路有限 PD 和 12.5 μs MuJoCo 原生步。地面实际分块碰撞，缺口 x 1.201–1.272、y −0.885–−0.844、深 0.024，均为仿真原生长度。单个槽在右前足的脚道内。体重归一化载荷用 W 表示。每次由同一个保存站姿开始，回合内自由根持续积分，外力、附着均为零。 [冻结身份](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/gap/CLEAR_FULL/RUN_IDENTITY.json)

## 17.25 秒匹配回合通过

| 实测节点 | 时间 |
|---|---:|
| 近侧右前足既有承重窗口结束 | 2.296 s |
| 右前足在缺口下探且实际失载 | 2.76309 s |
| 决策检测到预期有载、实际失载，触发收腿 | 2.765 s |
| 右前足抬出缺口 | 2.80210 s |
| 全足垫悬空过远缘，足垫中心轨迹始终处在该脚道的 y 范围 | 3.05124 s |
| 远侧上表面原生承重达 0.02 W 并持续 | 3.22715 s |
| 教师二次抬足后全垫安全退过近缘 | 4.140 s |
| 近侧重新测得 0.03 W，进入安全制动 | 4.925 s |
| 台架释放制动并再次允许前进 | 6.450 s |
| 撤掉推进意图，续行窗六足均完成周期 | 16.050 s |
| 末窗通过站稳检查 | 17.250 s |

远侧承重首刻足垫中心 x=1.31778，三个球垫的整体最小 x=1.30687，缺口远缘 x=1.272。承重持续 457.85 ms。关闭反射的同缺口对照在 7.595 s 首次直接承重于远侧同一脚道，晚 4.36785 s。同缺口保留制动与右中足指令、关掉右前足修正的对照，至 6.45 s 仍无远侧载荷。匹配填平场景在同处 2.788 s 有正常上表面支撑。首触发前的采样状态、决策、原生接触和足垫几何逐字节匹配。 [原生配对验收](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/gap/CLEAR_ACCEPTANCE.json)

首次失载后 200 ms 的其他五足合计最低承重 0.99482 W，五足各在 0.02 W 以上。右中足在触发前已有 0.24169 W 支撑，因此本回合的角色是已承重足维持支撑。换足补撑仍使用独立回合里从卸载转入承重的判据。 [原生配对验收](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/gap/CLEAR_ACCEPTANCE.json)

同一 1,380,000 步连续回合含 9.6 s 续行，前向 0.07163 原生长度/s，最大原生滑移 0.48144，六足周期计数 LF/LH/LM/RF/RH/RM 为 1/2/2/2/2/2，重新起步前和末窗原生站稳均通过。全程最低质心高度 1.25458，非足支撑峰 0，原生外力标志 0。独立复验的决策、轨迹、原生接触和足垫几何存档字节一致。 [续行原生验收](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/gap/CLEAR_FULL/NATIVE_AUDIT.json)

## 远侧保持和交接的对照

- 固定保持远侧锚点的 6.45 s 回合独立完成 516,000 原生步。末 200 ms 右前足足垫最小 x=1.30135，原生上表面承重持续高于 0.07101 W。总足部支撑最低 0.99999 W，最低质心 1.27389，非足支撑为零，站稳通过。 [远侧站姿原生验收](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/gap/FAR_STAND_AUDIT.json)
- 直接在远侧落地后交回普通站立策略，会把右前足拖返缺口。将锚点直保持到 6.45 s 后直接交回普通行走原型，同一错误在 50 ms 内重现。原始直接交接的远侧踏足通过，但随后的站稳、续行、滑移均失败。修订教师明确先抬高趾垫、全足越过近缘，再下放，并在站稳后由台架解除制动。 [直接交接失败证据](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/gap/DIRECT_FAR_AUDIT.json)
- 施教者用试验几何确定落脚路径，用真实关节、足底载荷和足端位置推进各阶段。输入记录 `x` 本体 213 项、`local` 17 项当前局部感受、`command` 当前前向原生长度/s 和有符号角速度 rad/s，以及 42 路实际 Action。该路径是有阶段状态的物理教师。身体前馈网络训练、其他落脚侧或缺口几何迁移、跨颈控制和完整 D1–D8 待各自验收。

## 可复验入口

Python 为 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python`，单线程 OPENBLAS/OMP/MKL 和 `MUJOCO_GL=egl`。下列命令在本实验目录内运行。仿真进程退出码为 0 表示写出实际物理记录，行为由验收 JSON 单独判定。

| 记录 | 命令 | 退出 |
|---|---|---:|
| 候选与独立复验 | `./drive_safe.sh clear CLEAR_FULL --native` 与 `./drive_safe.sh clear CLEAR_REPEAT --native` | 0/0 |
| 同缺口关闭反射 | `./drive.sh off QUAL_OFF --native` | 0 |
| 同缺口只关闭右前足修正 | `./drive.sh brake_only QUAL_BRAKE_OFF --native --duration 6450` | 0 |
| 同位置填平 | `./drive.sh flat FILLED --native --duration 4200` | 0 |
| 远侧锚点短回合 | `./drive_safe.sh hold FAR_STAND --duration 6450 --native` | 0 |
| 配对验收 | `python -B audit_crossing.py CLEAR_FULL --repeat CLEAR_REPEAT --off QUAL_OFF --filled FILLED --brake-only QUAL_BRAKE_OFF --output CLEAR_ACCEPTANCE.json` | 0 |
| 远侧保持验收 | `python -B audit_far_stand.py FAR_STAND QUAL_FAR FAR_STAND_AUDIT.json` | 0 |

初版完整回合 `OFF/FAR/FAR_REPEAT` 缺少评估器需要的 `walk_ms`，退出 1，保存可读轨迹；改正后 `QUAL_OFF/QUAL_FAR/QUAL_FAR_REPEAT` 各自实际重跑、退出 0。首个长回合因同一报告器缺陷终止，第二个因其保留已证明会失败的直接交接终止，退出各 143。早期视频和失败几何交接的时序独立保留。

同次物理轨迹配对视频逐帧解码 636 帧，已实际检查关键帧。橙色标记近缘，青色标记远缘。 [同次视频](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/gap/VIDEO_CLEAR/浅缺口远侧复踏配对.mp4) · [关键帧](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/gap/VIDEO_CLEAR/浅缺口远侧复踏关键帧.png)

## 延时行走失败与步态查找表诊断

第一次将原型续行从 9.6 s 延到 45.6 s，完成 4,260,000 原生步，仿真退出 0，验收退出 1。原生滑移 0.49733，末段站姿通过，前向均速降到 0.02256，低于 0.05 门槛。质心最终 x=0.31575，远缘 x=1.272，末态只有左右前足位于远缘 x 线之后。前 16.05 s 与已验收短回合原生记录一致。 [延时失败审计](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/gap/CLEAR_TRANSIT_AUDIT.json)

离线原样复算原型动作，持续前进的每条动作逐元素复现。2,880 项有限姿态表在 20.845 s 锁定最后一项 2879。25 s 起维持同一项，最大动作绝对值降到 3.29e−7，质心平均前向 −7.96e−13 原生长度/s。同期台架推进输入始终为 0.071713。 [查找表锁定复算](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/gap/STALL_DIAGNOSIS.json)

## 长行走接触锚点与地形拼缝

以低凸台已实证的世界坐标接触锚点，在 6.45 s 冻结后的实测稳定站姿上接续有界关节控制。计划前向峰值 0.113 原生长度/s，续行窗 33.6 s，最后 1.2 s 站稳。右前足的实体缺口范围和深度沿用。碰撞体、外力、运动关节范围沿用冻结输入。

- 第一版接触锚点和独立复验均到 34.128 s，保存 2,730,240 原生步及逐字节一致轨迹，实际终止门触发。质心首次越过 x=1.272 于 25.165 s。随后左后足在 y≈1.026、实际槽的另一侧接触地块拼缝。32.056 s 同一足端有上平面和成对的 ±x 横向法线，32.076 s 三个垫的接触均卡入拼缝侧面、足垫底已至 −0.04671，32.094 s 非足支撑超过 0.01 W。解析使用原回合保存姿态重建接触对，原生轨迹保留独立。 [X 向拼缝诊断](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/gap/X_SEAM_DIAGNOSIS.json)
- 将同一缺口的四周平面改按 y 排列，短探测、长预测的前段和填平对照都在 2.177 s 停止，早于右前足失载。右后足在 y≈−0.873 处落入拼缝。恢复用原始 x 布局，保留原始实体浅缺口和站姿快照。 [Y 向实体槽失败](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/gap/Y_COURSE/RESULT.json) · [Y 向填平对照](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/gap/Y_FILLED/RESULT.json)
- 仅提高贴近拼缝的摆脚和重选锚点的 X 布局复验，两次均到 34.102 s、2,728,160 原生步，左后足最终 z=−0.77672，非足峰达 0.57257 W。原始这组脚道通过证据保持有效，长程验收仍失败。 [离缝步态失败实测](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/gap/X_SEAM_COURSE/RESULT.json)

## 41.25 秒全身通过和末段站稳

最终给左足区域 y≥0 使用一整块连续上平面，右侧保留原槽地块、开口坐标和深度。消除了左后足先前卡入的 x 向接缝。控制沿用有界接触锚点和读出的真实地形边界，冻结身体和首次站姿一致。新同几何的无反射对照、填平对照和独立 6.45 s 前缀验收通过。右前足仍在 3.22715 s 远侧承重，随后安全抬足返回近侧站稳，台架于 6.45 s 解除制动。 [更新地面上的局部实证](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/gap/HYBRID_FULL_LOCAL_AUDIT.json)

33.6 s 续行中，质心在 25.12206 s 越过 x=1.272。右后足在 34.407 s、左后足在 36.764 s 各首次确认在远缘之后持续承重。行走到 40.050 s 撤令，41.250 s 站稳。末态质心 x=2.85437，六足全部 18 个球垫的最小 x=1.73151，分别超过远缘 1.58237 和 0.45951。末 200 ms 各足最小原生支撑均高于 0.10034 W。此处用实际右前足脚道的实体槽验收复踏，用其远缘对应的 x 参考线验收全身质心、所有足垫和支撑。 [全程原生验收](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/gap/HYBRID_TRANSIT_ACCEPTANCE.json)

主回合和独立回合各 3,300,000 步，执行决策、轨迹、原生接触、足垫空间和原生验收报告逐字节一致。实际续行前向 0.10614 原生长度/s，滑移 0.27717，最低质心 1.25323，六足各 6–7 周期。全程非足支撑和外力为零。续行后右前足在槽底和侧壁达到 0.002 W 的原生步均为零。原生工程门和末窗站稳全部通过。视频使用保留物理状态渲染，773 帧已完整解码，已查看保存关键帧。 [同次全程视频](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/gap/浅缺口通过全程.mp4) · [同次关键帧](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/gap/浅缺口通过全程关键帧.png)

新增复验命令与退出码均为 0。物理完成时长、实际行为和记录一致性由验收另行审计。

- `./drive_hybrid.sh course HYBRID_COURSE --native` 及同命令目标 `HYBRID_COURSE_REPEAT`。
- `./drive_hybrid.sh course HYBRID_LOCAL --native --duration 6450`，`./drive_hybrid.sh off HYBRID_OFF --native --duration 8750`，`./drive_hybrid.sh filled HYBRID_FILLED --native --duration 4200`。
- `python -B audit_hybrid.py HYBRID_COURSE HYBRID_OFF HYBRID_FILLED HYBRID_FULL_LOCAL_AUDIT.json`。
- `python -B audit_transit.py HYBRID_COURSE CLEAR_FULL HYBRID_TRANSIT_ACCEPTANCE.json --repeat HYBRID_COURSE_REPEAT --local HYBRID_FULL_LOCAL_AUDIT.json`。
- `python -B render_transit.py HYBRID_TRANSIT_ACCEPTANCE.json 浅缺口通过全程关键帧.png --video 浅缺口通过全程.mp4`。

这轮的可复验产物是固定条件下的工程教师和物理证据。教师读取试验槽和拼缝位置，续行由台架解除制动。新的前馈模型权重、不同缺口几何和腿相位的泛化、完整 D1–D8、跨真实颈部闭环仍各自待验收。
