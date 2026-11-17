# 父审阅后的有限教师执行入口（本文件不是执行结果）

当前6个Python入口与PREREG已ONLYCODE准备，子代理没有导入数值库、读取NPZ数组或创建Core。`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/pivot_teacher_design/RUNNER_CODECHECK.json` 是当前源码哈希与标准库检查记录；早先DESIGN中“没有runner”描述是上一准备阶段，现由本文件及RUNNER_CODECHECK接续。

父先review源码，再用已资格环境串行执行。以下Python路径沿父实际资格选择；示例为本地已有venv_body。每条命令新进程，不能复用预检中已经改变私有目标的teacher。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/pivot_teacher_design/preflight.py --execute-parent-preflight
```

预检构造恰好一个Core，不调用tick/step。它检查71不透明mesh、42有限PD、无附着/额外力、7身体mesh体长、旧warm接口、新teacher.action(0,.005)不改真实状态、私有actualframe水平Jacobian中心差分与0原生步。此时Core处初始站姿、尚未settle，检查的是接口与坐标，不是支撑或pivot行为。保存PREFLIGHT包含旧identity完整XML/资产/依赖与编译数组指纹。

实际预检通过且sourcehash一致后，父托管启动一次：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/pivot_teacher_design/run.py --execute-parent-threecases
```

固定顺序stop、left_pivot、right_pivot；每个freshCore依次1s原settle、4.8s原warm、9.6s新算法教师，随后独立saved-only audit。最大3×1,232,000=3,696,000原生步，0神经forward/训练。没有自动加档、续训或覆盖重跑。JOBS.json记录6阶段、命令、起止、日志和exitcode；物理/审核实现异常停止依赖pipeline并保留证据。合法unsafe提前停止保存实际窗口与FAIL任务，不把它当数值实现错误重跑；审核处理0或1条候选记录的短窗schema。

`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/pivot_teacher_design/trials/<condition>/` 将保存IDENTITY、INITIAL_STATE、RNG、BODY_LENGTH、HANDOVER、完整trace_1ms、teacher_samples_5ms、geometry_1ms、逐决策teacher_diagnostics、RESULT及VERIFY。旧Trace/PeakRecorder逐节点AST提取，保留原数值代码；避免整个student_wave的Torch导入。source身份在启动/结尾复核，所有实际导入本地模块和XML资产由旧identity捕获。

新pivot周期仅测loaded≥.1W且padz≤.004连续10ms→unloaded≤.05W连续5ms且净空>.015→重新loaded10ms，timeout4s，无前向advance门；warm最后10ms可用于初始承载armed。runner流式计数与audit独立批扫描分别实现，标准库假样本已验证完成/无抬升/单候选行三边界一致。停步周期不适用、瞬时速度阈值未冻结，因此停步task_pass_candidate始终null，不凭低漂移宣布停步已通过。

旧worldX≥.05与旧净位移slip照原公式计算、保留所有FAIL；新pivot门另列yaw≥45°、max COM drift≤.5体长、支撑均值≥.9、COM≥1.2、原生NF0/额外力0、完整9.6s与六腿实际pivot周期。Body速度均值/RMS/峰值按体长每秒、逐腿承载切向速度/体长每秒及载荷积分另报。早停的完整窗口门必须false。

保存后audit不创建Core、不导physics/MuJoCo/Torch：独立从数组重算yaw（含实际root四元数交叉核对）、漂移、身体速度、周期、旧五门、动作时间对齐和有限PD力矩上限，复核所有产物/源hash与identity规范摘要。原生每子步极值依据实际PeakRecorder摘要，审核不会重新求解物理；未保留全原生接触数组，不冒称完整D1—D8新验证。完整71mesh可见元数据保留，本批尚不生成视频，渲染应从同次保存状态独立做。

本阶段仅判断工程教师有没有可学的实际停步／原地转向轨迹；不加载新脑，不声称经neck控制，不把教师clock/desiredyaw/worldanchors交给部署ANN。若物理未过，先看IK饱和、支撑缺口、真实换足与yaw/漂移，不追加盲训练。
