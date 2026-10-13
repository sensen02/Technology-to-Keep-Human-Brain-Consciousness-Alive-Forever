# R28：执行 A——锁守卫、无锁规范模型与全状态平衡复测

性质：手工实现、测试中的仿真原型，不是成品。这里的运动学、力和激活结果均来自 MuJoCo，不是真实果蝇行为记录；项目使用的公开连接组数据另论。

## 已完成的模型修正

- 新增 `/run/media/sensen/Data2/cell_wound_prototype/tools_test_standing_locks.py`。旧规范模型 exit 1，列出七个有效 RF 单关节 equality；新规范模型 exit 0。合成注入验证改名锁、默认不激活的锁也会失败。
- `/run/media/sensen/Data2/cell_wound_prototype/tools_stand_and_ground.py` 在首次编译组装模型时即执行锁守卫（写入规范模型前），并在后续修复和最终模型处复查。此次构建退出码 0。
- 规范产物 `/run/media/sensen/Data2/cell_wound_prototype/outputs/muscles_six_legs/fruitfly_six_leg_standing.xml` 已无继承锁；nq=49、nv=48。SHA256：`65d3904dce9802f46997a92cb3f80cd13c032f847c1e1f6710694d65cfdd73b6`。
- `/run/media/sensen/Data2/cell_wound_prototype/outputs/stance_solution.json` 新增模型路径、哈希和锁守卫结果，哈希与规范 XML 一致。早期守卫加入后再次构建，XML 哈希保持不变。
- 旧 XML、旧姿态报告及旧全量检查报告保存在 `/run/media/sensen/Data2/cell_wound_prototype/outputs/rebuild_r28/before/`。
- 双向性守卫：42/42 关节双向，246 个非零肌肉—关节力矩项；工作区间守卫：90 条肌肉、0 条关键帧带外、满激活时 0 条零力。通过不代表站立。

守卫范围：针对继承的单关节腿 equality，包含默认关闭的定义；不证明所有潜在 weld/connect/tendon/多关节约束都不可能限制运动，也不证明站立。全状态诊断另要求模型完全没有 equality。

## 全状态诊断取代根自由度诊断

新入口：`/run/media/sensen/Data2/cell_wound_prototype/diagnostics/diag_full_state_equilibrium_r28.py`。
旧入口 `/run/media/sensen/Data2/cell_wound_prototype/diagnostics/diag_static_equilibrium_activation.py` 明确停用并退出 2，不再产出“相对残差小所以已找到平衡”的结论。

新诊断：
1. 按关节类型和地址覆盖所有 48 个速度自由度，不假设 qpos 与 qvel 索引相同。
2. 平移绝对阈值 1 mm/s²、转动绝对阈值 1 rad/s²，激活动力学阈值 1e-8/s；都是预先声明的工程阈值，不是生物学标准，可通过参数修改。
3. 每次目标函数调用新建 MjData，清空速度、外加力和 warmstart；显式映射 actuator_actadr，并设置一致 act/ctrl。
4. 控制下界为 1e-4，优化遵守控制/激活边界。不能把旧测试中“命令为零”直接叫“肌肉纯被动力”。
5. 单独报告瞬时平衡与每积分步释放观测；释放包络包括高度、身体触地法向载荷、倾角、线速度和角速度。
6. 自检抓住“根加速度零、腿仍加速”和“相对残差下降 99%、绝对仍超标”的伪通过；锁定模型在优化前被拒绝。

第一组：初始激活为允许区间的 0.1，2000 次目标调用（含数值微分）预算。
- 规范化全状态残差 L2：11183.17 → 3294.09。
- 最大分量为绝对阈值的 1719.898 倍，48/48 分量均未通过。
- act_dot 最大绝对值为零；全新数据重复计算 qacc 差为零。
- 优化停止原因：预算耗尽。结论是候选失败，不是不可行性证明。
- 0.5 s 释放最低胸节高度 0.7397 mm、峰值倾角 49.20°、峰值身体接地法向载荷 1.33547 倍模型体重；未保持声明包络。
- 这里连平衡点都没有确认，不使用“找到不稳定平衡”的说法。

第一组数据和已查看图像：
`/run/media/sensen/Data2/cell_wound_prototype/outputs/rebuild_r28/full_state_equilibrium.json`
`/run/media/sensen/Data2/cell_wound_prototype/outputs/rebuild_r28/full_state_equilibrium.png`

## 基线与待完成复验

此前全量落盘报告实际为 60 PASS / 1 FAIL（接口文档过期），不是 61 项通过。接口符号比较无变动，已重生成文档并确认 stale 检查通过。

第二组：初始激活为允许区间的 0.5，预算 10000 次，实际调用 5369 次；优化器以 ftol 收敛、success=True 退出，但 48/48 加速度分量仍未通过绝对阈值。残差 L2 29878.05 → 3235.07，最大归一化分量 1693.93。0.5 s 释放最低高度 0.74947 mm、峰值倾角 49.82°。说明优化器成功标志也不能替代物理平衡判据；两次局部搜索均未找到合格候选，不是不可行性证明。数据与已查看图像分别为 `/run/media/sensen/Data2/cell_wound_prototype/outputs/rebuild_r28/full_state_equilibrium_start05.json` 和 `/run/media/sensen/Data2/cell_wound_prototype/outputs/rebuild_r28/full_state_equilibrium_start05.png`。

本轮完整运行结果：62 项中 61 PASS / 1 FAIL，检查器按 `--exit-nonzero-on-fail` 正确退出 1。唯一失败为 force_downgrade_audit：新全状态诊断使用接触法向力但未带显式降级说明。已补充该诊断的力通道声明（只作模型内部诊断、不作生理载荷结论、不挪用 stack A 旧载荷比），并独立复跑该审计：PASS，3/3，包含失败注入；新诊断自检再次通过。

保留完整失败报告 `/run/media/sensen/Data2/cell_wound_prototype/outputs/rebuild_r28/all_checks.json` 与修正后定向报告 `/run/media/sensen/Data2/cell_wound_prototype/outputs/rebuild_r28/force_downgrade_recheck.json`。准确表述是“全量 61/62，唯一失败已修复并定向复验通过”，不是伪写一份同批 62/62。物理平衡与释放试验仍失败，和回归通过无关。

## 证据边界与遗留问题

- 构建器零命令释放最终高度约 0.692 mm、过程最大倾角约 47.4°，仍不能站立；这是零命令且致动开启，不是严格零激活。
- 构建器旧姿态报告的若干几何字段来自不同构建阶段；其中 final_up_dot_up0 实际来自 1.8 s 采样，不应和 2 s 高度拼成终态。当前哈希一致只证明文件身份，不解决这些字段的测量语义。新全状态诊断逐步 forward 后重新测量，独立于旧字段。
- R27 报告补了更正：相似有限时刻终点不能证明同一吸引子；不能从后来发现的锁推断 R5–6 的历史锁状态；零命令不等于纯被动肌肉；无肌肉力下变体相同不能排除肌肉工作时修复作用。特别地，“0.0002 mm”只是指定变体/指定锁状态/指定时刻的差，不是一般修复价值或单一几何因素的因果量。
- 规范重建后，R26/R27 旧诊断硬编码当前规范路径并要求七个锁，不能原样重跑；必须使用保留的历史模型，不覆盖历史证据。
- stack A 与 stack B 的质量、控制和记录不可混用；此次 stack B 质量为 2.49427 mg，并非 README 里 stack A 的 1.02431 mg。
- 下一步应分析无锁模型上最大关节残差与可达力矩、接触/姿态耦合；此次局部优化失败不足以证明所有姿态/激活都不可行。不切换备用方案。
