# 前足绊脚抬腿教师记录（S2/S5）

新实验目录位于 `/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T060654Z/teacher`。沿用三球足、71 个身体 mesh、42 路有限增量 PD、同一名义冻结积分态，决策 5 ms，执行器目标更新 1 ms，原生 12.5 μs，0.5 秒静立、9.6 秒步行、1.2 秒停稳。仿真平台使用机械接触，无附着和额外外力。

## 碰撞和感觉

用冻结原型平地已保存的身体姿态对小盒做离线几何筛选。正式场景 A 仅在同一平地添加一只静止低盒，原生长度尺寸 0.018×0.016×0.29，中心 [0.86,−0.735,0.145]，约 0.10 体长高。全部既有质量、惯性、执行器和碰撞属性保持并核对。RF 胫节和近端两个跗节有碰撞的原有 mesh。每腿的脚垫法向荷载为旧指标。新传感记录三个指定段与环境的物理接触力在胸部坐标的三分量，关节试验只使用实际反向分量触发。全部脚及腿段保持原世界碰撞，盒接触单独记录包括错误段，原地面接触观测保持原口径。

还保存目标足实际垂直等方向速度，以及由其他 35 个当前目标在冻结标称库匹配得到的 RF 关节目标偏差。后者由当前执行器状态产生，可提供离开障碍后的修正信息。教师看到 RF 反向力 >0.002 W 且该足地面荷载 <0.08 W 后才开始生成偏差动作。首轮正式选择纯 RF 7 路执行器限速纠正，其余 35 路继续冻结原型按各自当前输入计算。教师用实际触发时刻来定位约 0.29 秒的抬腿和回落曲线，运行时学生接口为当前身体状态和触力。

## 运行和验收

以相同物理形状的原型为有障碍配对臂，以原冻结原型无障碍为另一对照。覆盖实际盒碰撞、是否只有具名段触及、经释放后稳定重落脚、其余脚承重、地面支撑与滑移、继续完成原六足周期以及起止站立。要求出现触力后量到的垂直净空较有障碍对照增加至少 0.020 原生长度。本试验障碍宜为局部反射复现，因此保留基线能自行清除低盒时的对照，落脚力、碰撞持续和冲量分开实测。所有长回合保留原生接触阵列及按 1 ms 对照的端点。完整 D1–D8 另行计算。

同身 Python 用 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python`，设置 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl`。本目录 C 接触探针由 g++ -O3 -ffp-contract=off -shared -fPIC 以同一 MuJoCo 3.9.0 header 和 library 编译。下面命令从本目录运行，省略相同 Python 路径。

```sh
python -B probe.py
python -B scan.py
python -B physical.py control CONTROL_PEG_A --scenario PEG_A.json --native
python -B physical.py teacher TEACHER_PEG_A --scenario PEG_A.json --plan PLAN_SINGLE.json --native
python -B physical.py control CONTROL_PEG_B --scenario PEG_B.json --native
python -B qualify.py TEACHER_PEG_A CONTROL_PEG_A TEACHER_PEG_A/TEACHER_ACCEPTANCE.json --absent ../student/FLAT_FROZEN
python -B replay.py TEACHER_PEG_A TEACHER_PEG_A_REPLAY
```

首轮短试 DEV_PEG_A 产生完整短物理轨迹但运行中协调修正脚本命名导致源码身份漂移，退出 1。保留当时记录。DEV_TEACH_A 由泛用模块名误导入旧教师，退出 1，无试验步。DEV_TEACH_B 触发时目标路径超出归一化速率域，退出 1；修正方式是在教师送出动作前按原域限幅，底层执行器限矩保持。短试 DEV_TEACH_A2、DEV_PEG_B、DEV_TEACH_B2、DEV_TEACH_A_SINGLE、DEV_TEACH_B_SINGLE、DEV_TEACH_A_LOCAL 完整 3.45 秒，工具及每臂日志退出 0。A2/B2 时钟仅可用于带解耦的开发教师，其余腿速率的变化使它们离开纯单腿比较，未入库。保存 B 试验形状，其宽 0.036，高 0.302，候选的另一形状评估保持 A 的训练。

平地冻结对照由训练代理独立运行。官方教师保持地面和静止盒物理，只记录所选实际触力，碰撞不会以脚本注入。

## 同体教师结果

A 冻结原型、A 抬腿教师和 B 冻结原型三项完整回合的 Python 退出码均为 0。四个统一条件含训练代理无障碍对照，均通过 904,000 原生步的地面工程检验。A 两控制同一冻结姿态，碰障前毫秒快照及决策一致，其同形场景中确实只有右前第二跗节触盒，原生初触为回合 2.38595 秒，教师首次介入为 2.390 秒。RF 地面载荷为零，其他五腿触发后 250 ms 的原生最低地面支撑为 0.99525 W。A 原型和教师都在 2.789 秒达到持续 20 ms 的重新承重，完成其余行走和最终停步。峰值足垫净空由 0.110497 提高至 0.179365 原生长度，三维脉冲阻力积分 0.000134228→0.000132756 W·s，碰障原生步数 873→877，地面滑移 0.482149→0.483233。该结果建立了方向性触觉触发的额外净空。控臂已能走过本轻障，抗撞击强度的证据仍限于这一轻障。

名义盒 A 物理教师资格 14 项全通过，包括与有障和无障两对照直到碰障前逐字节匹配、单腿纠正、其他腿支撑及重新承重。此回合向学习代理交付 2,260 条决策，只有 walk=1 的修正样本用于拟合。B 控制保存同身体有障情况下的留出配对参照，B 不加入训练。

- [教师入库资格](TEACHER_PEG_A/TEACHER_ACCEPTANCE.json)
- [原生配对统计](PAIRED_TEACHER_STATS.json)
- [教师和冻结原型视频索引](../student/VIDEO_TEACHER/VIDEO.json)

已查看教师配对关键帧的碰障前、初触、较高摆足、重新承重及末端站立。

以保存教师动作启动全新 Python 进程，独立 904,000 原生步重放退出 0。2,260 条决策中的每一项当前观测及局部状态和原始教师一致，统一毫秒轨迹、决策和压缩原生接触证据均逐字节一致。见 [独立物理重放核验](TEACHER_PEG_A_REPLAY/REPLAY_AUDIT.json)。正式教师、同盒 A 对照、同盒 B 对照和独立教师共 3,616,000 原生步，其余按开发短测记录。验收文件、几何与体态锁定核验完成。依照原定义的完整 D1–D8 和更大障碍物尚待独立方案。
