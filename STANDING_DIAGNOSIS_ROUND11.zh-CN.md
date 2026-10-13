# 第 26 轮：撤回接触独因归因，定位右前腿锁定回归

手工实现与测试的仿真原型，不是成品，没有真实果蝇数据。

## 本轮复核推翻的结论

1. 上轮“肌肉关闭”仅将 muscle gainprm/biasprm 的第 0 列设零，未真正禁用力。重测 act=1 时仍有最大 actuator_force=77.5864、qfrc_actuator=13.0036。因此基于该对照排除肌肉并归因接触模型的结论撤回。
2. 当前规范模型有七个有效的右前腿 joint equality locks。上轮重建镜像模型时未传 --unlock-locked-legs，把一腿源资产的锁重新带入六腿模型。61/61 通过没有覆盖这项条件，不证明模型可站立。
3. 独立代码审查发现静态平衡求解只检查六个自由根 DOF，使用相对残差改善而非全状态绝对阈值；3.87% 不能证明平衡，更不能证明不稳定平衡。此结论暂撤回，等待修正诊断后复测。
4. 上轮 N4a2 将第一帧相等改为 1.5 倍增长，没有验证时序或缓存基线一致性。这不构成充分因果对照，需重新做匹配的 intact/lesion/sham。

## 本轮可复现的 2×2 对照

脚本：/run/media/sensen/Data2/cell_wound_prototype/diagnostics/diag_lock_ablation_r26.py

同一规范 XML，每组新建 model/data；仅切换七个 RF equality 的活动状态及 ACTUATION 禁用标志。每个禁用组在全部积分步骤核查 actuator_force 与 qfrc_actuator 峰值均严格为零。时长 0.5 秒。

| RF locks | actuation | 最终倾角 deg | 最终胸节高度 mm |
|---|---|---:|---:|
| 7 个有效 | 正常 | 46.0021 | 1.40219 |
| 7 个有效 | 真正禁用 | 46.1986 | 1.41005 |
| 解除 | 正常 | 9.6197 | 0.72571 |
| 解除 | 真正禁用 | 18.3594 | 0.71792 |

结论仅限这些条件：锁定对约 46° 的倾斜有重大影响；无锁条件下肌肉也影响倾角。接触模型独因归因不成立。解锁后仍下沉，未站住。

## 已实施的修正与范围

- /run/media/sensen/Data2/cell_wound_prototype/tools_stand_and_ground.py：组装时移除名称/引用均匹配 RF、无 joint2 的单关节 *_locked 约束，不移除无关 equality。
- 已生成独立变体 /run/media/sensen/Data2/cell_wound_prototype/outputs/muscles_six_legs/fruitfly_six_leg_unlocked_r26.xml；未替换规范模型。2 秒被动释放仍下沉，最终高度约 0.692 mm，不能算站立。
- /run/media/sensen/Data2/cell_wound_prototype/diagnostics/diag_passive_tilt.py：改为 ACTUATION 标志禁用，逐步验证力归零，移除接触独因打印结论。旧说明和产物仍需全面更新；本轮不声称全量套件复验完成。
- /run/media/sensen/Data2/cell_wound_prototype/engine/embodied/force_channel.py：就地撤回接触独因归因。原 stack-A 的 NON_PHYSICAL_LOAD 数值降级不受此撤回影响。

## 接下来

在无锁独立变体上重做全状态平衡与激活可达性检验；绝不把残差相对下降称为平衡。补构建锁定守卫、失效禁用注入、匹配时序观测对照后，才允许推广规范模型。备用方案暂未启用。
