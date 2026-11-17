# 视觉身体v2：6基本条件＋2视觉对照，父执行说明

当前仅CODE_ONLY，标准库实际20项通过、exit0，0数值模块/数组/模型/forward/Core/渲染/训练。所有核心源停止修改。旧body1源/代码检查/失败视觉候选不改。本工具只有视觉候选2严格final500离线全部所需目标与独立审核、saved500审核PASS后才能绑定，未知SHA或失败时拒绝，不自动绕门。

## 父唯一资源下准确顺序

实际视觉候选2全部通过后，取真实最终SHA：

```bash
python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/visual_body_v2/prepare.py --bind-completed-offline-pass --visual-sha256 ACTUAL_VISUAL_POLICY_V2_FINAL500_SHA
```

prepare仅标准库核实际权重/证据/全部sourcehash，绑定新ROOT与新loader，保留原运动用户接受/strictFAIL；无Core。执行后已存在bindings/freeze时拒绝覆盖。

实际0Core预检：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/visual_body_v2/preflight.py --parent-resource-released
```

加载新视觉与冻结运动、临时model别名严格恢复、20msrelease/5ms持帧与batch1真RGB编码。新增0Core固定帧查询测试：实际rawrelease继续，actor保持captureANN4.78的已发布左帧。无新渲染/物理。

父managed顺序执行8回合和独立审核：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/visual_body_v2/orchestrate.py --parent-resource-released > /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/visual_body_v2/PIPELINE.log 2>&1
```

JOBS/CURRENT_STATE动态记录逐阶段PID/log/exit，不纳冻结来源。已有JOBS或partialtrial应先查实际结果，不盲重跑/覆盖。全部重任务由父拥有并收集。本子不执行。

## 新范围与最小复用

单AST适配body1原直接loop，完整newruntime globals；不改旧源，不叠十层source替换。显式变更：queue.query改actor_query、neural添加rawreleased与actorused字段、control副产物入RESULT、独立审核actorID/clock容差/controls、budget8。模型/物理算法/指标保持。clock从两种整数表示相乘得到时间允许1e-10表示误差，实际神经输出重放仍严格exact。

6基本条件：V_empty/V_center/V_left_pulse/V_right_pulse/V_left_to_right/V_right_to_left，目标在5.8s实际头部基准一次锚定世界，前.8/左右±.4/正前0；柱z1.2、高2.4、半径.12。on.6/off3.6/reverse4.8，真头相机随body反馈，不从静态库播放。

2控制条件预先固定：

- V_left_occluded：左柱从.6一直存在，3.6加入固定世界宽墙；目标＋墙的真RGB进入actor，同时间另渲染同墙无柱作为counterfactual，存真实NPZ/IDs/SHA/MAE/exact。只在captureANN3.6—3.8且release≤3.82内首次真完整遮挡帧建立恢复起点，要求恢复窗持续采样遮挡且turnRMS≤.05，真实遮挡不足则precondition false，不后选晚时刻。
- V_flip_fixedframe：场景左目标4.8换右、实际相机/20msqueue始终采新图；actor在4.8冻结最新已发布左帧（全程capture_id529，captureANN4.78/release4.8），其16features保持，32GRU/256motor连续。raw released IDs继续变，actor-used IDs冻结，真实场景与图像变化另诊断。此对照不要求turn为0或跟真实右目标，不算新导航成功。

时序例子：普通脉冲环境source在ANN3.6消失，该次真实capture3.6于3.62才release给actor，正常恢复评价窗是ANN4.12—4.62（以首个已发布off帧再加.5..1秒），不是旧气味4.1—4.6。on.6也需.62才发布，response从actualused首visible帧起算。遮挡控制同理以实际完整遮挡帧release时刻为起点，上限3.82；将来视频应分别注明环境事件/capture/release，不能混用。

正常6与控制2判定分列，未以控件测试替代真实视觉响应；原滑移FAIL/完整周期/NF0条件保留，视觉失败可能是statichead到movinghead分布变化，不能用手写ifvisible/距离门/归零reset补造结果。

## 输出

每trial仍保存770实际双眼RGB、CAPTURES、INITIAL_STATE、15400身体端点、2880神经决策、完整288memory与独立cache16。新决策字段raw_released_image_capture_id/raw_released_capture_time_s/raw_released_release_time_s与image_capture_id/time/release(真实actor所用)分开，actor_frame_frozen显式控制标志。所有时间/坐标/控制标签仅环境或审计，CNN只有RGB输入。

V_left_occluded另有occlusion_counterfactual_rgb.npz，键rgb和capture_ids，RESULT对应artifactSHA；独立audit重算真实pixelMAE/exact并与CAPTURES标签绑定。每capture仍证明qpos/qvel/ctrl/qacc/qacc_warmstart/qfrc/efc_force/contactpipeline渲染前后不变，不额外mj_forward。

NORMAL6_SUMMARY.json为基本6条件独立判据结果；VISUAL_BODY_SUMMARY.json汇总normal6_all_functional_pass与control2_all_pass及各失败项，最终all_declared_targets_pass仅两者都通过。无视觉行为权重准入之前不会产生这些运行结果。
