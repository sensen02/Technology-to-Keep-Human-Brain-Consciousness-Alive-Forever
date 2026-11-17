# 固定私有影子根的左右 pivot v2：ONLYCODE，物理可行性未知

本目录只准备两个新完整身体教师回合，不重跑stop或任何旧pivot。唯一控制因素是将私有FK根／初始方向／初始原点保持在5.8s交接实际姿态；yaw±.15rad/s、1s ramp、lift .12、slot .8s、landing lead2s、步序、原IK/支撑/有限PD、初态和seed全部沿用v1。

依据实际文件 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/stop_shadow_control/MATCHED_COMPARISON.json:3-50,176-227`，固定影子stop与旧actual-shadow stop配对检查全部true，固定影子完成9.6s支撑，1msCOM最低1.275870037、nativeCOM最低1.275868132、NF0、最大COM漂移9.163948e-5体长、yaw变化−.001357926°；stop_speed_PASS仍null。`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/stop_shadow_control/SUMMARY.json:3-11`确认只新跑1,232,000步、没有重跑旧控制、没有ANN/优化器。这个证据支持该stop下私有影子策略的作用，不证明pivot或神经控制已成立。

teacher.py继续通过绝对路径加载v1原始教师，调用同一make_teacher(c,left/right)构造全部原始目标，仅覆盖action为直接原StaticTeacher.action；fixed_action函数AST必须与已实际资格的stop_shadow_control/teacher.py内对应函数完全相等，adapter在每个入口检查。实际qpos/qvel/外力均逐动作检查未写，私有根／框架不得改变。新真实躯干仍自由运动，固定的是私有FK推算参考，不是实际根。

runner/preflight/audit三个入口只用一层AST直接复用v1，防止stop包装器再包一层的隐式依赖。数值常量序列严格相等，仅结果kind、CLI左/右和预检探针元数据改变。新8核心源、旧v1全部源、stop控制源及实际proof、两方向旧RESULT/VERIFY和完整保存artifact均入frozen，实际identity继续补原Core/XML/资产/编译数组。科学文件不改；父新渲染图不作为运行输入。

预检1Core/0integration：保留v1全模型/体长/私有FK水平Jacobian中心差分和actualstate不变检查；同Core独立构造原actual与新fixed左pivot教师，比较t=0首action/q_des/desired_motor_target精确相等，不复用任何probe教师进正式运行。stop实证JSON、所有来源SHA、fixed_action AST一致性是准入条件。右pivot由相同factory反号参数产生，本预检不是两方向实际行为证明。

新两回合各1s旧settle+4.8s旧walkingwarm+9.6s固定影子pivot，最大2,464,000原生步，0ANN、0训练。完整记录／早停与旧原生峰记录不变。新任务门保持：实际有符号yaw≥45°、maxCOM漂移≤.5体长、支撑均值≥.9、COM≥1.2、nativeNF0/额外力0、六腿实际卸载抬起重承载和完整9.6s。旧forward≥.05与净位移COM滑移原样单列，不当pivot门。请求yaw不是实现yaw；短窗或失败如实保留，不自动扫参数。

每方向saved-only audit先执行原v1独立指标审核，再对其同方向旧actual-shadow归档比较trace前5800ms所有字段、geometry前4800行、sample前960决策bitexact；首候选动作、HANDOVER真实状态/目标/缓冲hash、初始RNG、编译参数、PD和体长均须相同。旧完整物理不重跑。任一配对不符则候选不准入；若新教师失败，失败本身不是计算审核错误，也不能证明身体不可能。

父先read源码；使用已资格Python环境，示例命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/pivot_fixed_shadow_v2/preflight.py --execute-parent-preflight
```

实际预检通过且父当前heavy任务结束后，唯一托管命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/pivot_fixed_shadow_v2/run.py --execute-parent-twocases
```

run串行left物理→left保存/配对审核→right物理→right保存/配对审核，共4阶段；每阶段JOBS/log/exit完整，无重试覆盖。产物PREFLIGHT.json、trials/left_pivot与trials/right_pivot内轨迹/原审核、MATCHED_left_pivot.json、MATCHED_right_pivot.json、SUMMARY.json。所有输出必须等待实际证据，不提前写PASS。

本子仅完成标准库语法/命名空间/数值AST/fixed_action与来源stopproof文本检查，没有数值库导入、NPZ数组、Core构造或物理。即使教师物理通过，也只能作为后续经neck2DN独立任务量ANN学习的工程教师数据；不存在外置CPG部署或新脑已控制pivot的结论。
