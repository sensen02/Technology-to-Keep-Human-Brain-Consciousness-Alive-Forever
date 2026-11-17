# 固定四例教师辅助训练数据采集：仅代码准备（S2/S5）

根目录：`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T221508Z/teacher_assisted_collection`。

永久标签 **TEACHER_ASSISTED_DATA_NOT_NEURAL_ACCEPTANCE**，范围 **BodyANNcomponent_NOTnewbrainneckdeployment**。这是有限教师辅助数据采集工具，不训练权重，不将辅助稳定记作神经自主成功。旧789f通过模型不覆盖；第二版007d四项行为失败及全部保存审核证据保留。主tracker由父统一维护。

已读最新 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T195245Z/CURRENT_STATE.json`。当前007d学生实际walk滑移1.228/LH无周期、stop速度RMS.0322高于.01、左右unsafe提前停；不能由拟合改善或保存审计通过推断行为成功。205真实预检和206真实14710教师查询已完成，原206仅最后报告导出缺json导致exit1；原查询不重跑、不修冻源。新prepare必须等待并检查独立report recovery的实际VERIFY/AUDIT/报告及全部source/artifacthash，保留原exit1 REPORT_EXPORT_ONLY身份。

四例顺序与命令固定：old_walk_control[1,0]、assisted_stop[0,0]、assisted_left_pivot[0,1]、assisted_right_pivot[0,-1]。各1秒settle＋4.8秒旧教师预热＋9.6秒collection。前200决策不NNstep，960次warm仅prime_old[1,0]旧256/new96恒0。5.8秒真实状态、actuatortarget、40ms缓冲、previousaction与352状态连续，不reset。完整71opaque/free-root/42有限PD、flatworld、native12.5us、无adhesion、无实际根写入或额外力，原unsafe100ms连续COM<.8或NF>.5W保持；异常与短窗保留、不重试。

old_walk_control是冻结789f旧256神经控制的实际匹配正控制，不是007d神经成功，也不同于旧完整336感知脑颈基线。本例学生命令[1,0]投影为旧模型合法[0,0]；实际动作直接来自m.cached_features内部m.old.step产生的oldmean_direct.tanh。完整007d352仍在真实观察上连续shadow。另存oldmean_compat=mean−residual，禁止用该浮点减法回读值伪称直接oldmean逐位一致。

common.shadow_details只调用一次m.cached_features，取directoldmean/oldafter，再用实际新模型模块ALPHA/BOUND和相同encoder→leakyGRU→head(cat(h96,currentencoded96))公式计算完整新均值/状态。没有doubleadvance旧state。预检和独立NN审核分别调用m.old.step与m.step_details，从相同beforestate验证直接旧均值/旧链bitexact、完整新均值/残差/状态≤1e−12。父预检明确记录真实非零NN调用，不能把其神经工作算0。

三个assisted例仅在真实5.8秒handover各创建一次合格fixedshadow教师，初始化真实足anchors/privatefixedroot/q_des并采用原qualified order_indices。每5ms在真实当前qpos/qvel/target上查询，同一教师私有q_des/history/phase持续推进；不沿理想旧轨迹查询，不每tickreset。实际施加固定.75*teacher_action+.25*tanh(完整007dmean)，再原bounded_action/actuator有限PD；全collection不调beta、不扫描、不暗转全神经。监督target保存未混合完整teacher_action。clock/anchors/privateq_des/mask只作教师标签诊断，不进actor；不加载脑/眼/颈，不造AN/DN。

标签采用StageB相同11项保守mask：实际preaction COM≥1.2、saved1ms NF0、query IK≤.01、unbacked计划载荷≤.01W、unsupportedfalse、qdot/rate/target/tau无clip、有限42及查询前后七真实状态完全不变。warm/settle/control的teachermask不适用而全false；control directold监督另标。全部无效行保存，mask不得遮盖物理失败。teacher NF判定是1ms实际源记录，不冒称nativeforce全峰；native峰另录。

保存完整或实际短窗trace1ms、settle后geometry、所有决策actualX238/u2/352beforeafter/directoldmean/compatoldmean/studentmean/residual/studentaction/teacheraction/mixedaction/fulltarget/mask/source，source0settle/2warm/7assisted/8oldcontrol。teacher JSONL逐行保存完整私有q_des/desiredtarget/IK/clip字段/plannedload/actualcontact和query前后七statehash。另保存INITIAL/RNG/runtimegeometry/HANDOVER和WARM_BOUNDARIES，记录first/lastwarm与handover真实target/buffer/神经状态及factory七数组前后不变。

固定12主阶段：4phys＋4freshNN保存审交替执行，再3teacherlabel独立零积分恢复审，再summary。每物理例≤1232000原生步，总≤4928000。教师标签审核独立3Core但0积分/0NN；预检单Core0积分。最大新Core为8，其中真实轨迹4个；没有自动扩展。若阶段执行错误，保存失败与partial停止，不重新查询或换例。行为unsafe提前停止是完整记录的失败结果，仍做保存审，不补1920行。

NN保存审核 fresh strict007d先load后任意物理后端，直接旧均值/旧256链独立逐位、完整new生产均值/状态/残差≤1e−12；核实际混合、监督目标、hold/source/clock/shape/hash/identity/PD/外力/标签schema。另核旧独立metrics实现得到的原五门与六腿周期工程描述。oldcontrol通过也只归冻结旧256；assisted工程安全/周期数据不归全神经。

teacher保存审核是单独无神经进程，复用已实际205/206验证的standalone restore.py而不调用旧query.main或改旧INPUT_BINDINGS。在离线Core恢复newtrace[ms−1]、真实target/buffer[ms−40:ms]，仅mj_forward重解约束；单persistentteacher从同一handover连续重算action/q_des/desiredtarget/IK≤1e−6、contacts和mask精确、query前后七数组相同；完整诊断也比较。恢复重解与源接触的差异明确记录，不声称nativeforce逐步重放。step与NN traps分别确认0，标签审失败不得把数据称合格。SUMMARY始终保留data_qualified_for_training=false，父须另做数据质量裁定才可另立有限DAgger。

所有旧science文件不改。新common仅单层compile旧原common中通用physical_stack/Trace来源/Peak/旧warmconfig/测量等helper，函数在新globals创建，旧原common SHA硬固定。新actionpath为实际独立实现，不声称metadata-only或numericunchanged。STD检查覆盖所有代码AST/新globals/源hash/actor没有teacherfield/原门与mask来源/固定时序预算及父审核接口；本子无Torch/NumPy导入、无NPZpayload、模型/Core/训练/物理/渲染。

父准确标准库检查命令：

```bash
/usr/bin/python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T221508Z/teacher_assisted_collection/static_check.py
```

父确认report recovery实际通过与源冻结后，依次执行下列命令。使用父既定实际20261110T125712Z/env环境，不使用另一个.venv替换套件身份：

```bash
/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T221508Z/teacher_assisted_collection/prepare.py --bind-completed-final600 --sha256 007d0c74118715eacd03d49c7e6847d8349e55db2793dd9d461a4b9c311e0fcd

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T221508Z/teacher_assisted_collection/preflight.py --execute-parent-preflight

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T221508Z/teacher_assisted_collection/run.py --execute-parent-fourcases
```

末条由父单一managed CPU1 job执行、完整收集12stage与原生数。当前未绑定、未实际预检或采集。未来人工脑→真实DN2来源队列→神经decoder2及完整感觉回归仍是独立必需阶段，当前采集工具不构成成品架构部署。
