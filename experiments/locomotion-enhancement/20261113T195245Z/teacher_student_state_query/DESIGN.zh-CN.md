# 阶段B：保存学生真实状态上的教师查询（ONLYCODE）

状态：**LABEL_QUERY_ONLY_NOT_PHYSICAL_SUCCESS**。本目录只准备离线恢复及算法教师标签可行性查询，子端没有加载数组／模型、导入数值库、创建Core、forward或物理。所有旧165/182/175/178/186来源、权重、结果与失败保持冻结；没有Dagger训练或新候选。

## 实际档案与索引可行性

已用标准库读取两cohort实际RESULT/VERIFY/INITIAL_STATE/HANDOVER和ZIP NPY头。首版454581...学生stop/L/R各1920候选决策；二版007d0c74118715eacd03d49c7e6847d8349e55db2793dd9d461a4b9c311e0fcd stop1920、左494、右776，分别保留原完整／早停长度。先复现已资格146stop与150左右教师各1920，之后查询6条学生轨迹，共14,710次教师action；main最多9个freshCore（3+6），preflight另1Core。walk两条只用保存oldmean对NN动作作代数对照，不生成walking教师标签，也不重跑NN。

精确索引来自 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/physics.py:139-164`：每1ms先增量target更新，再80原生子步／末forward，然后buf追加target、t加1、pad导数更新。Trace行i是(i+1)ms末。因此preaction ms=5800必须取trace[5799]，此时NN在5.8s计算的动作尚未作用，下一步才是5801ms。每个query的target历史是trace.target[ms−40:ms]，不是把初态或0重复40次。

真实INITIAL_STATE确有qpos49、qvel48、ctrl42、target42、40×42 buffer、xfrc/qfrc；HANDOVER含这七机械字段的显式hash。恢复必须对该case自己的HANDOVER七项一致。若一般初期ms<40需要拼接时，只可用已记录initial.buffer+实际target；ms1需要initial.padz而本数据没有它，helper会拒绝，当前全部query≥5800不受影响。缺字段或非零外力max不能复原完整向量时一律报错，不造0。

## 离线恢复的含义

restore.py把保存qpos/qvel/ctrl/time、actuator.target_rad、buffer及相关1ms元数据复制到独立Core；_padz_prev来自当前末行，_pad_rate由两个末padz差/.001恢复。实际外力每行max==0时能严格证明全数组为0；不是凭假设填零。sensory.features为当前纯函数，不需要捏造神经感觉history。

这里的qpos赋值是**离线保存状态重建**，不是新真实运动或root轨迹控制。随后只运行mj_forward重新计算当前几何、bias、passive及接触约束；qacc/qacc_warmstart等未保存求解器历史不会伪造，其接触／载荷／NF／速度／actuator_force与原档案的差异单独报告，不冒同次原生力重放。机械七数组与时间／target／buffer恢复必须逐位精确，模型compiled arrays及PD参数用原IDENTITY验证。

step陷阱禁止mj_step/mj_step1/mj_step2和Core.tick/tick_rate/tick_policy/reset/reset_from。Core构造时允许原resetDataKeyframe/forward，因为不积分；之后只允许本目录restore。Torch若因旧物理依赖已经存在，只patch已加载torch.load及Module._call_impl，绝不为本任务加载NN权重或执行forward；若晚导入使前段覆盖不全，计数标UNKNOWN而不是伪0。上下文退出恢复所有patch。

## 参考教师与顺序状态

stop用 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/stop_shadow_control/teacher.py`，左右用同根 `pivot_fixed_shadow_v2/teacher.py`；均直接沿用已通过实际身体试验的固定私有FK策略。每case恢复5.8s后构造一次，捕获该实际handover足锚点／固定私有根／q_des；后续每5ms按保存学生preaction状态恢复当前身体和target，教师私有q_des、clock、anchors连续，不每tick冷启动。顺序来自旧实际PREFLIGHT.order_indices，参考slot/lift/yaw/ramp/lead不改。

preflight和main先在三个教师自己的保存轨迹上重放建议action，与原Y42、原JSONL私有q_des／desiredtarget／IK／contactcounts逐项比较。明确保存max差，不假bitexact；门为action／私有q_des／desiredtarget最大差≤1e−6，coldhandover七hash和水平Jacobian中心差分≤1e−4。接触／载荷重解差异报告但不冒原生接触精确。三个资格不过就不继续学生query。

## 每行标签资格，不是恢复成功

所有行均保留。标签mask跑前固定：实际保存before COM≥1.2；before源1msNF=0（不是未记录的native逐时峰）；query IK最大误差≤.01原生；unbacked计划载荷总和≤.01W；teacher.unsupported=false；qdot/rate/target/前馈tau_ff无裁剪；有限42动作；恢复exact。另保存当前姿态下desiredtarget／nexttarget预测的原始PD力矩及超原生cap标志，明确它们是局部代数预测，不是新物理实测；不偷偷将这些附加描述变成跑后筛选门。unsupported、关节限位／不可达或算法异常都标无效，缺标签用NaN数组及独立错误记录而不是0动作。无效行不删除，不事后改阈值挑数据。

输出每case query_action42、原NN/教师实际action42与差、私有q_des42、desiredtarget42、实际target/ctrl、IK6、qdot/rate/tau及裁剪、计划足载、真实重解contactcounts、源COM／1msNF与yaw、mask及理由。有效mask只说明该局部教师建议满足本工程可行性条件，未证明实际施加后能恢复；今后actor输入仍只允许真实physical112/old256/task2/new96，教师clock／anchors／rootstate只作标签求解与分析，不作隐藏actor输入。

## 教师所需寄存器与当前actor的访问差异

实际读取BASE/sources/proprioception.py:10-18,38-43确认：0:42 joint-offset=(q−q0)/halfspan；42:84 target-error=(target−q)/halfspan；84:126关节速度/10；126:168 target-position offset=(target−q0)/halfspan；后续为足载／倾角／身体速度。183:195是clearance补充；195:237是上次动作；237为旧.075常量。两类target特征经[-10,10]感知裁剪并float32化，**126:168不是effort**。

models.py:50,113-116的physicalmask排除42:84、126:168、195:237；因此old256递归编码与新112通道encoder没有这两个当前body积分寄存器的直接输入。然而models.py:140-142的旧base均值仍显式包含 `−gain()*obs[...,126:168]`，新整体mean继承该冻结prior。不能概称整个旧／新motor完全不读target，更不能据此证明不可观测或身体任务不可能；原行走曾通过。直接encoder的autograd梯度为0只描述该被测分支／冻结detach，不能推出wholemean的数值functional independence；整体仍有上述显式target-position输出prior。本任务不额外执行NN干预验证，只说明源码依赖。

每行另保存actualtarget、querydesiredtarget及实际源X的两个target归一化片段，描述其与queryaction−实际NNaction的相关／误差。教师rate法则本来直接用target，这种相关不证明唯一机制。未来若需显式body-local target/efference观测，必须另立候选和版本；读真实当前body寄存器不是脑绕颈，但教师privateq_des理想目标不得冒充当前感觉。本阶段不改任何mask或网络。

## 资源与父执行顺序

父先阅读，再独占CPU托管预检：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T195245Z/teacher_student_state_query/preflight.py --execute-parent-query-preflight
```

恰1Core／0积分，恢复旧left教师5.8s，核handover／模型／Jacobian，顺序查询前20步。按实测20步耗时×14,710／20×1.5再加前置耗时余量估计main≤300秒才READY_QUERY_300S；RSS1800MiB。若预计超限先BLOCKED，父须在真正main之前另立预算，不运行后延期或自动缩数据。

预检actual all_passed后唯一主查询命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T195245Z/teacher_student_state_query/query.py --execute-parent-saved-teacher-query
```

main 3教师资格+6学生Core，walk2无Core，最多9Core、0原生积分、0NNforward、0优化更新。300秒与1800MiB全流程资源门，source/RNG/compiledPD前后校验。输出queries/各npz+JSON、STAT.json、REPORT.zh-CN.md、VERIFY.json。任何旧科学输入不写；有源/恢复/资格错误保留产物，不重跑已完成物理。

代码ONLYCODE检查不等于恢复／标签资格的实测结论，真正数值重解差异与有效标签比例均待父运行。运动目标继续active，不把教师查询或档案审核当作新身体成功。
