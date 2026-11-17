# 首批气味刺激A/B核心实现交接

状态：核心与零Core预检完成，物理与独立逐trial审核由父托管唯一heavy orchestrate_v2.py执行。当前无本子活动作业；本报告不宣称任何A/B实际行为通过。

## 已实现

1. 原motor COMBINED789f严格loader与原sensorbfd6严格common loader，sensor-only身份守卫在物理后端导入前实际执行，不篡改旧guard/资格函数。
2. 新神经投影[relu(turn)/3,turn]，256运动记忆+32行为记忆=288；2受体filter/pending环境状态另列。受体5ms causal pending使用上一完整区间响应。
3. 同原1s稳定+4.8教师预热+9.6ANN连续回合，暖机行为状态初始保持，不对ANN输出施加刺激标签动作；A仅显式高层命令探针，B行为真实读取受体，不强置无气味turn为0。
4. 原生每步非足峰/步数/extra汇总，完整15400端点、2880神经行与头部几何，运行中所有实际source/hash/compiled绑定；新试验runner无actualqpos/qvel写入，无handover reset。
5. 新进程audit.py与最小audit_v2 wrapper：motor均值/逐42shapeTorch动作/256状态、行为命令/32状态、完整288连续链、受体因果filter/pending、samplelocalXYZ、保存姿态yaw、交接hash、source/runtime、原全部五门和六腿周期独立重算。
6. 纯stdlib orchestrator逐个spawn physics→audit，A3全部functional+integrity通过后B6，否则A_GATE=false/B_NOT_RUN。每个子process PID/日志/退出码写JOBS，不常驻Torch模型；completed result+audit身份正确可复用，不重跑已完成物理，partial目录明确拒绝盲重启。
7. B功能评分匹配真实B_noodor，原fiveengineering/30deg端点指标/用户视觉接受/刺激功能效果分开，弱刺激不要求满30°，原历史FAIL不改写。所有时序、门、n=1/condition已预先登记；B汇总新文件，不回写trial RESULT。

## 实际预检与身份

子managed bash53 exit0已收集，真实20checks PASS、0Core/0积分/0训练、2.71s：两个冻结权重严格加载、head AABB算offset与localXYZ exact、neuralReLU耦合、288初始形状、离线9.6s确定性/因果pending、neutral全部2880均值和256hidden对原基线逐位相同、行为拒绝第三输入、动态文件排除。

FROZEN.json SHA256 `5f0752ff7dd6d1f43c8685bf37577a6f360f37c7a5ddcf5f4526e091dd2407bd`。

预检后、任何物理前，父静态复核要求补baseline VERIFY和计划中有效受体起算latency。原20check预检及6个核心v1源码/FROZEN保持不变，新AMENDMENT_01_BEFORE_ANY_PHYSICS.json和轻量v2 wrappers只增补输入身份与summary时钟/统计措辞，不改物理计算。12项stdlib补充检查exit0，0模型导入/0Core；FROZEN_SUPPLEMENT.json SHA `086f6025e7f946012a99c9d85af5996b66d0d034cf9c15aced69338291b4f190`。baseline原neutral RESULT/audit同789f SHA、resultsha与全部checks绑定，trace/neural/geometry哈希核对。

浓度左强对固定 `[0.4510848855472039,0.34581507678322543]`，右侧交换、等强取均值；来自旧HEAD审计几何、前.6/角+.12/sigma.5/amplitude1.25静态高斯源，0噪声首批新声明，不是移动空间导航。离线pulse实际行为turnpeak.9998867，只是输入路径检查，不是身体响应结果。

所有数值源码已冻结，CURRENT_STATE/JOBS/报告/visual不纳身份；任何后续代码修改必须新范围/新文件，不改已captured旧输入或VERIFY。

## 父托管启动入口

`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T224234Z/orchestrate_v2.py`

输出schema：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T224234Z/trials/<ID>/` 含RESULT/VERIFY/IDENTITY/HANDOVER及trace_1ms/geometry_1ms/neural_decisions/runtime_geometry。

ID：A_left_half、A_right_half、A_switch；全部通过后B_noodor、B_equal、B_left_pulse、B_right_pulse、B_left_to_right、B_right_to_left。

神经键：time、X238、command2、raw_command2、raw_turn标量、hidden_before/after256、behavior_hidden_before/after32、full_hidden_before/after288、mean42、policy_action42、actually_applied_action42、control_source、receptors2、concentration2、filter_before/after2、pending_before/after2。实际键为`command`和`raw_command`，上文尾数字仅表示维度。

RESULT保留checkpoint789f/behavior_checkpointbfd6各SHA、identity_sha256、transition、native、autonomous/cycles/yaw、functional。engineering_pass仅原五门；strict_endpoint_qualified含30度历史端点口径另列，endpoint30_applicable_to_stimulus=false，old_motor_qualification_unchanged=true。

审计必须allchecks true，且full288_hidden_chain/actual2880_motor_outputs_exact/actual2880_behavior_outputs_exact/causal_receptor_filter_exact/actual_sensor_geometry_exact/metrics_cycles_recomputed等强检查。B功能终判在独立B_FUNCTIONAL_SUMMARY.json，原trial functional待匹配不回写。

视频目前PENDING父实际渲染与全解码/目视；C/D空间闭环、训练、到达/摄入均NOT_RUN。
