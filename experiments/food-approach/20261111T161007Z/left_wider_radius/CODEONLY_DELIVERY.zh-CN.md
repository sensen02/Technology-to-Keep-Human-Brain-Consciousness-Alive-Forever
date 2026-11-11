# LEFT 更大转弯半径候选：CODEONLY 交付

本次仅完成S4/S5新候选代码与CPU1纯NumPy预检。未运行物理、训练、渲染或子agent。旧源码/结果/VERIFY/tracker未修改。父负责后续顺序物理。本说明不宣称候选物理通过。

预登记：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/left_wider_radius/PREREG.json`。

固定一个LEFT+.06候选：finite映射保持，step.8/rest/pose/PD/lift.12/order/全部原门保持。原4.8s直行预热仍为stride.36。之后1s用b=smoothstep同步过渡：stride=.36+.12b、q=stride/.36=1+b/3、v=.075+.025b、omega=.06b、k=omega/v、kdot=.0045*bdot/v²。参考phase=t不变；水平delta按q缩放，delta_dot=q*basevelocity+qdot*base_delta，再进入有限SE2解析速度。原z/velocity_z/stance/source clock保持。所有目标只作teacher监督，无根或物理姿态写入。

`command_speed_yaw`记录名义v/omega，command2=(v/.075-1,omega/.06)，forward新域0..1/3明确只作诊断、未来ANN尚未资格确认。旧X238中的target_speed=.075只保留为历史特征常量，不声称是新actual desiredspeed或推进门。原worldX≥.05、slip≤.5、COM≥1.20、support≥.90、原非足≤.20W、strict post nativeNF0、原body-X advance周期all6、实际yaw≥30°、无早停全部保持。额外报告每腿载荷加权足速分子、载荷分母、COM位移速度分母，不能只凭ratio下降声称绝对滑移改善。

主要源码完整路径：

- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/left_wider_radius/turning_teacher.py`
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/left_wider_radius/diagnose.py`
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/left_wider_radius/audit_wider_radius.py`
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/left_wider_radius/nonfoot_pairs.py`
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/left_wider_radius/source_identity_wider.py`
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/left_wider_radius/preflight_wider_numpy.py`
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/left_wider_radius/freeze_wider_sources.py`

新诊断默认CODEONLY，必须--run-physics才运行，且仅固定单LEFT；显式ROOT、factory与继承源码路径检查，不调用父sweep。复用已冻结父one/PeakRecorder，新增本地recordersubclass只在原生solved NF>0时read实际contacts。按最多64geom pairs保存观测计数、first/last time、geom/body ID及实际model accessor名称、最高worldZ力/位置/时间与post peak；溢出只留count和一个peak，无完整native数组。无非足时dict为空。实际MuJoCo accessor仅未来run执行，codeonly未导入MuJoCo。

新审计不调用或导入任何旧verification writer。只用exclusive方式写新目录`AUDIT_WIDER_RADIUS_<UTC_pid_ns>.json`，绝不写旧VERIFY/RESULT。除了原门/逐腿周期重算/yaw/commands/sourcehash，还要求实际新trace前5800个1ms样本全部字段与旧完整straightprefix bitexact。数学prefix检查不冒称已核验未来physicalprefix。native peak仍是捕获摘要，不是完整独立原生重建或D1—D8。

冻结：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/left_wider_radius/SOURCEHASH_WIDER_RADIUS.json`，1511项源码/实际旧身份依赖/资产/参考/旧完整轨迹与旧VERIFY输入。SHA256 `665360c769dcde42a43968b24252ea4eb7858cdb05ed17f2ff71f784e99b28a9`。旧物理IDENTITY每个输入的原hash也在freeze前验证；strictfreeze不提供覆盖/自动重锚选项。新报告、日志和后续audit不列作执行输入；旧捕获VERIFY只读。

预检：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/left_wider_radius/PREFLIGHT_WIDER_NUMPY.json`，23/23 PASS，7源码AST/PREREG JSON解析PASS，所有实际命令exit0。CPUaffinity[1]、OPENBLAS/OMP/MKL1；无physics/mujoco/torch/static_teacher/student_wave导入。

- 整个t≤4.8sample精确等于原stride.36直行参考；全2880样本非水平字段exact。
- steady参考等价直接stride.48/k.6映射；实际rest取旧REFERENCE并与旧trace relativefoot[999]精确比对，无Core/FK。
- qdot/kdot解析速度360时点中心差分，最大误差3.6682736848092645e-9，tol3e-6，h1e-6s。
- 明确检查4.8/5.6/5.8及其后步/周期接缝：水平位置一阶外推残差max1.135924687645229e-12，速度两侧差max3.2512202184087347e-6（两侧间隔2e-6s）。无目标跳变。
- ramp5ms目标位移max.011829182803570098，小于连续解析速度包络.011919058939323181+1e-5。
- 五stance脚距检查通过；LM首诊断摆窗参考正X最大.11618332634496126，超过原.03门；只是数学目标余量，物理可行性NOT_RUN。
- 非足归因纯records检查：零、64上界/溢出、计数/firstlast/peak/postpeak均通过，没有构造物理或contacts。
- 默认诊断CODEONLY、无结果audit明确NOT_RUN；最后复核1511输入hash未变，未产生新physical目录或audit文件。

父可顺序执行的命令（本子任务未执行）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/left_wider_radius/diagnose.py --run-physics > /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/left_wider_radius/RUN_WIDER_RADIUS.log 2>&1
```

随后新进程独立审计：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/left_wider_radius/audit_wider_radius.py > /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/left_wider_radius/AUDIT_WIDER_RADIUS.log 2>&1
```

失败如实保留，不自动扫参、不改周期口径。源码冻结后不再修改；本子任务已交付命令并停止，无后台job/临时缓存待清理。未来实际运行状态由父单独记录，本文件描述CODEONLY交付时点。
