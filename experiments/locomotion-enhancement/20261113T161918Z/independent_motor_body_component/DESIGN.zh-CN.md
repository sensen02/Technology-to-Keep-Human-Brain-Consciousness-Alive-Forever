# 新motor四项身体组件诊断：ONLYCODE，行为与最终权重身份未知

本目录准备walk[1,0]、stop[0,0]、left_pivot[0,1]、right_pivot[0,-1]四项明确实验命令。它们是新身体ANN组件能力诊断，不是人工脑→关节绕颈成品，不声称当前三部分模型已部署这些任务。只有组件身体资格成立后，才另做脑NN→来源真实DN2队列→ANN decoder2接入及完整感觉回归，不能省略。

当前calibration最终SHA尚未由本目录绑定；prepare拒绝未知/格式不对的SHA，须实际final600的TRAIN/RESULT、3阶段JOBS与STAGE_VERIFICATION、saved0/300/600各新进程重放VERIFY、EVALUATION及独立EVALUATION_VERIFY、对应artifact SHA全部成立。binding固定format、actualSHA和状态[128,64,64,96]=352；不修改已运行的校准源码、旧789f或教师记录。

模型严格装载发生在任何MuJoCo/physics/FlyGym导入之前。身体保持原full71opaque、42有限增量PD、无附着/额外力、自由根、原生12.5µs。固定seed20261113。1s settle不运行NN；4.8s原teacherwarm每5ms调用m.prime_old(x,[1,0],state)一次，仅old256前进，new96始终零，实际动作仍由旧teacher控制。5.8s连续交接检查实际qpos/qvel/ctrl/target/buffer/force及352状态不重置；自主9.6s每5ms仅m.step_details(x,currenttask,state)一次，tanh(mean)通过原42有限PD，不再调用教师，不把输出乘速度置零、不写根姿态/轨迹/外力。

旧obs238最后的.075仍是历史常量特征，不能借此偷偷实现速度；任务只进独立command2，其他特征没有case/time/phase/地形坐标。当前没有AN/DN接口，不制造sourcequeue/port字段；没有相机任务，不造770眼帧或视觉缓存。记录全部1ms15400身体状态与受力，14400行settle之后几何，以及3080条5ms记录（含200settle、960warm、1920ANN），明确mean/oldmean/residual/352beforeafter/command/policy/actualaction/source。oldmean按生产mean-residual同一实现保存和重放，避免另调用旧motor推进两次。原生每子步NF/COM/额外力峰由旧PeakRecorder保留，未冒称全原生接触数组或D1—D8新验证。

新门在任何实际身体NN运行前冻结，且不追改旧教师：

- 共有：完整9.6s、足载均值≥.9、COM最低≥1.2原生、原生非足峰=0、额外力=0。
- walk：旧原五门全部成立（worldX forward≥.05、旧slip≤.5等）、六腿旧实际行走周期完成。
- stop：maxCOM XY偏离≤.1体长，交接后首.5s过渡之外的身体XY速度RMS≤.01体长/s，实际yaw绝对变化≤5°；不要求摆腿周期。RMS=sqrt(mean(vx²+vy²))，速度由连续COM差分/1ms，再按实际Thorax旋转投影，缺少.5s后样本不能按0通过。
- pivot：沿用实际有符号yaw≥45°、maxCOM偏离≤.5体长、六腿真实承载→卸载/净空→重新承载和共有门；不把请求角速度当实现值，不把低漂移称严格零位移。

各case仍照旧公式计算forward和净COM位移slip并保存原始失败；停止/pivot不使用趋零COM分母作为新滑移门。逐腿承载切向速度/体长每秒及载荷积分另报。体长从共同settle后warm前的七个准确Head/Thorax/abdomen编译mesh沿Thorax轴投影范围计算，排除腿翼/眼触角，SI未知。

安全终止沿用连续100ms COM<.8或NF>.5W；原生异常、NaN、身份漂移即停止。提前停止的实际窗口保留，不补满、不重跑或无限候选。四回合最多4,928,000原生步，0新训练。

fresh NN保存审核须重新严格装载同一actual600，然后以真实保存X/command重播全部3080决策：settle无step、warm960prime、ANN1920step，原始mean/residual/oldmean/352状态/action逐位一致；1ms实际动作与5ms输出对齐。独立重算yaw/漂移/身体RMS/支撑/COM/NF/周期及每任务门，复核原始identity、source和artifact哈希。审核通过仅说明记录一致，不能自动称行为通过；结果按各门实报。

父完成calibration实际资格并read源码后，先metadata绑定（下列REAL_FINAL600_SHA256必须替换为实际已审核SHA，不可猜）：

```bash
/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/independent_motor_body_component/prepare.py --bind-completed-final600 --sha256 REAL_FINAL600_SHA256
```

随后父独占资源托管零积分预检：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/independent_motor_body_component/preflight.py --execute-parent-preflight
```

严格final600先载入，恰1Core/0integration，核352接口/旧冻结参数/8张量实际Adam600/全身71mesh与42PD、第一次warmprime、四命令mean42/state352、有界残差、实际状态不变、纯测量observer无物理副作用。它不是行为验证。只有实际all_passed后启动一次：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/independent_motor_body_component/run.py --execute-parent-fourcases
```

4physics+4fresh NN saved audits+summary共9阶段，JOBS记录命令/起止/日志/exit，不自动重试覆盖。最终SUMMARY逐case并列新任务门与旧五门、实际时长及失败，继续标new_neck_deployed=false。未知最终SHA与全部行为结果等待实际证据，不提前写PASS。
