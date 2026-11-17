# S2/S5 平地运动教师运行记录

试验目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T045152Z/teacher`。冻结原型及台架来自 `/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T000536Z`。源形状资格、三个预定增益、冻结物理门、配对准入见 [预登记](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T045152Z/teacher/PREREG.zh-CN.md)。

## 启动物理身份

- FlyGym 使用本机 `venv_body/bin/python`、FlyGym 2.1.0、MuJoCo 3.9.0、`HybridTurningController`、`FlatGroundWorld` 和种子 17。五秒参考按 0.1 ms 积分、每 1 ms 存样。包源码、编译模型、控制器、足名和执行器顺序逐项写入 [无附着参考](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T045152Z/teacher/FLYGYM_FLAT_NO_ADH_V3/RESULT.json) 和 [附着参考](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T045152Z/teacher/FLYGYM_FLAT_ADH_V2/RESULT.json)。
- 同体教师用 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python`。每条回合登记 XML、编译物理、初态和控制代码哈希。地面平坦、足端无附着、42 路有限 PD 增量执行器。每 5 ms 决策，1 ms 汇总物理，原生步 12.5 μs。
- 保留项目原型的支撑控制与步态时序。在原型的净空窗口中，使用同名足的 FlyGym 实测垂直形状，经项目腿部雅可比求速率校正。四腿达到源曲线门。两条后腿继续由原型控制。摆动选段见 [迁移来源](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T045152Z/teacher/SWING_REFERENCE.json)。
- 每条准入示范保留决策前 213 维身体观察、当前步行意图和 42 路实际归一化动作。教师可用自己的参考时序。其动作仅经冻结身体的有限执行器施加。

## 命令与运行

所有命令显式设置 `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl`。命令从教师目录运行，`python` 对应上一节的解释器。

```sh
python -B flygym_flat.py FLYGYM_FLAT_NO_ADH_V3 --duration 5
python -B flygym_flat.py FLYGYM_FLAT_ADH_V2 --adhesion --duration 5
python -B flygym_flat.py FLYGYM_SPEED_0.08 --duration 2.5 --signal 0.08
python -B retarget.py --build
python -B retarget.py DEV_G30 --gain 0.30
python -B retarget.py DEV_G30 --gain 0.30 --register-existing
python -B verify_teacher.py DEV_G30 --native-replay
python -B verify_teacher.py DEV_G30 --recover-native-metadata
python -B retarget.py G30_lean_left --gain 0.30 --start lean_left
python -B ../../20261116T000536Z/prototype.py --out CONTROL_lean_left --start lean_left
python -B verify_teacher.py G30_lean_left --control CONTROL_lean_left --native-replay
python -B gait_features.py ../baseline/nominal_9600_raw DEV_G30 GAIT_FEATURES_FULL_WINDOW.json --project-settle-ms 0
python -B d1_d8_scope.py DEV_G30
```

另采集对称信号 0、0.025、0.20 的源校准，以及名义 DEV 增益 0.05、0.15。`moving_forward` 和 `uneven_feet` 按上述 `lean_left` 的成对命令运行。目录中同名日志记录实际结果，各命令以不同目录保存证据。

- 首次无附着参考因控制器仍返回六个关闭标志，无相应执行器时应用动作返回 1。修正为只调原位置执行器。一次参考长度计算发现静态融合链接索引为 -1，改用具名几何后两组参考均退出 0。早期片段和日志保留。
- 参考构建初版反复读取压缩数组，主动终止；第二版保存形状数组后 JSON 含 NumPy 整数而退出 1。下一版缓存所需数组并转换索引，退出 0。
- 名义 DEV 三组物理及轨迹完整，第二次向独占结果文件写元数据时返回 1。保留原物理结果，另写来源文件，核验保存的增益数组精确一致，恢复退出 0。后续同体运行直接写独立来源文件。
- 名义原生复跑完成原始数据写入，结束阶段散列函数收到字符串而退出 1。保留原生数据，重新计算门值，并核对独立记录的动作及轨迹哈希逐位一致，恢复退出 0。后续原生复跑直接生成核验清单。

## 结束与独立报告

名义增益 0.30 的冷站 0.5 s、直走 9.6 s、撤令后站 1.2 s 独立重放 904000 原生积分步。档案轨迹和动作重放逐位一致。向前速度 0.07171305 原生长度/s，最坏滑移 0.48138857，质心最低 1.25614691，足载均值 1.00000005 W，六腿周期和两端静立均合格，原生非足地面力与外加推进为零。 [名义验收](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T045152Z/teacher/DEV_G30/TEACHER_ACCEPTANCE.json)

同初态增益关闭的配对显示，七段可迁移窗口的足垫高度加权 RMS 变化 0.00550840 原生长度，向全部源形状目标的加权误差从 0.01952220 降到 0.01407926，四条迁移腿分别有正投影。相同观测下步行窗动作 MAE 0.00246147，512 个决策受影响。最大全动作差 0.48495，包含原型近邻切换。 [足端来源效应](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T045152Z/teacher/DEV_G30/SOURCE_EFFECT.json)

无附着源信号 1.00 的体长归一前速约 5.651/s，四只前中足可辨认步频约 12 Hz。项目约 0.025 体长/s，完整直行段同体教师步频约 0.208 Hz。FlyGym 信号 0.08 前速 0.0284 体长/s，共用清晰摆动代理在该条件下六足均为零。比较记录含每足的步长、承重比例和相位。本轮验证四足摆动垂直形状迁移，匹配速度的六足完整步态待验证。 [完整步态对照](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T045152Z/teacher/GAIT_FEATURES_FULL_WINDOW.json)

源效应在 `lean_left`、`moving_forward`、`uneven_feet` 三个启动样本均复现。0.5 s 稳定后各回合摆动响应与名义几乎相同，行走数据多样性有限。各回合的入库资格以自己的验收文件为准。D3、D5 需同回合辅助窗，D6 需原生质心速度差。 [名义 D1–D8 范围](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T045152Z/teacher/DEV_G30/D1_D8_SCOPE.json)

## 视频与交接

父代理用已保存配置渲染标称增益关和增益 0.30 同步视频，解码确认 566 帧并打开对照关键帧。将通过原生验收的记录独立交身体复训子代理。 [轨迹视频记录](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T045152Z/VIDEO_TEACHER_GAIN30/VIDEO.json)
