# 平地弧线出弯连续衔接 S2/S5

工程台架，原自由根及 71 个 mesh 身体，原 42 路有限 PD、三球足端、无附着。原生 12.5 µs 物理求解，动作间隔 5 ms，每个 1 ms 汇总现有本体与接触字段。测试初态来自旧完整站姿快照及冻结身份。原六动作保持旧目录。新增内容只在本试验目录，未接入人工脑或生物颈部通路。

## 物理控制与对照

首轮预登记站立 500 ms、冻结直走入弯 4,800 ms、弧线 9,600 ms、出弯及后续走动 6,400 ms、最后站稳 1,200 ms，全程连续。规划期前向速度 .13 原生长度/s、左右目标角速度分别 ±.115 rad/s，在旋转阶段末起持续 1.2 s 渐消。期望路径速度平滑过渡到 .091；在末尾用渐变制动并保持已落下的世界足锚。切线直走参考同时记录实际测量的出弯朝向。路程几何是教师的私有 IK 参考，外力和实际身体根始终和旧仿真版本一样由有效力学积分决定。六足按既有 .8 s 次序完成完整摆动，各自支撑期持续使用上一足锚。转向时的右后足净空使用旧左弯已知 .028 原生长度，右弯不施加这个旧左弯补丁。

本轮主对照为各方向在同一个锚足弧线前缀上，于原出弯时刻直接换用冻结直走。匹配比较前缀原生、决策和毫秒轨迹必须逐位相同。旧直接切换左侧 2.112、右侧 1.560 的配对以同一指标记录，来源 `/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T004235Z/FINAL_ACCEPTANCE.json`。新方法的脚端前瞻约 2.8 秒，未来出弯计划可能在前缀末段改变预期脚端目标。旧档案是独立历史参照，本轮同前缀消融另外明确给出。

## 验收窗口

- 弧线方向和转角、六足换步、平均足载、最低质心、零非足触地使用原 `turning.score` 检查。
- 入出弯后的首 4.8 s 与旧出弯可比。实际每个原生接触力及相应切向速度乘积按具名足积分，单足绝对值取载荷归一，最坏足再除同窗口水平净位移平均速率。阈值仍 .50。固定轴取当时身体朝向，前进至少 .05 native/s，净偏航绝对值至多 10°。全 6.4 s 出弯和实测航向的稳态直走窗口也各自记录。稳态净偏航绝对值至多 4°。
- 原生自由根、前进、脚载和总力学场与末端预登记门互不替代。全程记录任何外力为错误，保存原生状态和每 4,031 步独立接触计算核对。
- 教师只在正式原生控制及末端站稳均通过后提供学习标签。`DECISIONS.npz` 的字段是 `ms`、`x`（冻结身体当前传感 213 项）、`command`（前进 native length/s、带符号偏航 rad/s）、`action`（42 路标准有限控制）。信号与实际关节控制分别留证。
- 教师内部使用可规划相位、预定脚落点和测量航向补偿。方法性质是算法教师，不能由其直接宣称学习模型或生物运动控制已验收。完整 D1–D8 需相应另案。

## 实际命令及身份

解释器 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python`，MuJoCo 3.9.0，`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl`。效果和每回合运行代码见 `RUN_IDENTITY.json`，进程退出码保存在同名 `.exit`。

首版执行 `./drive.sh left TEACH_LEFT_A LEFT_PLAN.json`、`./drive.sh right TEACH_RIGHT_A RIGHT_PLAN.json` 和 `direct_ablation.py {left,right} DIRECT_{LEFT,RIGHT} {LEFT,RIGHT}_PLAN.json`。四个 22.5 s 连续积分均正常退出 0。教师两方向全门通过，直接切换配对出弯质量未通过。 `audit_pair.py` 两方向退出 0，核验到出弯开始的三类文件字段全部逐位相同。
首个 `certify.py` 准入读取了错误的 NPZ 字段 `poses` 而返回 1，随后尝试的重放也因该准入文件未建立返回 1。保留 `*.FIRST.log` 和 `*.FIRST.exit`。改为真实字段 `pose` 后原物理轨迹不变，两方向准入均退出 0。最终影片 1,209 帧全解码，实际打开了包括方向转换及终末的关键帧。录像只读取坐标渲染，不进行物理积分。
两条首版准入示范随后执行 `certify.py TEACH_LEFT_A LEFT_PLAN.json --out REPLAY_LEFT_A` 及右方向同名命令，均退出 0。独立完整原生重积分的三份 NPZ 逐位一致。含重放的 `audit_pair.py` 输出 `MATCHED_LEFT_WITH_REPLAY.json` 和 `MATCHED_RIGHT_WITH_REPLAY.json`。首版教师、配对直接切换与独立重放六组实际总计 10,800,000 原生步。
补充试验 `./drive.sh left TEACH_LEFT_SOFT LEFT_SOFT_TRIM.json`、`./drive.sh right TEACH_RIGHT_SOFT RIGHT_SOFT_TRIM.json` 各 1,928,000 步、两方向完整联合门通过、两个程序退出 0。已封存为供拟合的较缓航向校正样本，原 A 样本继续保留。 `SOFT_TRIM_EVIDENCE.json` 从原生数据独立给出后段确切 5.3 秒零偏航命令指标。旧、新教师配对和首版重放的所有物理求解合计 14,656,000 步。
