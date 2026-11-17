# 长历史行为恢复候选1：独立审核通过，功能验证未全部通过

父唯一托管bash-58已收集exit0，完整执行preflight、固定1000步训练、初始/500/1000三个新进程保存审核、最终12配置验证与独立验证审核。全部真实执行stage命令/PID/log/exit见 `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T234150Z/JOBS.json`。无物理积分、无运动模型加载与修改。

## 实际完成

- 24训练配置、12验证配置，每条1920×5ms=9.6s，静态工程双受体浓度，源/时间仅离线教师标签。
- 实际preflight53项全PASS，生产/脚本扫描/梯度/Adam最大误差6.245004513516506e-16；完整1920BPTT，无历史detach。
- 同原SteeringBehavior2→GRU32→turn架构，freshAdam1e-4，batch4，固定1000更新，训练用时约373s。未采用500择优。
- 最终权重 `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T234150Z/TRAIN/checkpoint_001000.pt`，SHA256 `4be0cd6a9d1f77e3be17323713f3fc8783adedd4b9e8a010df69922c7d1a09e7`。
- 最终独立审核 `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T234150Z/VALIDATION/VERIFY.json` integrity_all_pass=true；权威功能结果 `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T234150Z/VALIDATION/RESULT.json` all_targets_pass=false。

## 功能结果

12配置中的4个pulse配置均因撤除窗口RMS失败；其余8配置通过各自判据。两个无气味配置输入相同，不是独立重复，不能把配置数当统计重复。

| 条件/配置 | 规定撤除窗口RMS | 门槛 | 判定 |
|---|---:|---:|---|
| 左脉冲V0 | .0727739702 | ≤.05 | FAIL |
| 左脉冲V1 | .0800567840 | ≤.05 | FAIL |
| 右脉冲V0 | .0819718941 | ≤.05 | FAIL |
| 右脉冲V1 | .0900335354 | ≤.05 | FAIL |

全部有向条件的预登记末段符号与2s响应门通过，两向反转末段符号通过。无气味RMS .0044983，等强全程RMS .0059289/.0098949，均通过。脉冲末.5sRMS约.003365，已无首批旧模型left持续饱和；返回并维持±.05带.25s的诊断时间距offstart约.965/1.12/.99/1.00s。但规定offstart+.5..1s整窗口RMS仍超门，不能据末尾恢复良好把失败改PASS。

本轮仅说明“这项长序列训练候选去除了这组静态序列的长期饱和残留，但恢复时间窗未通过”，不证明唯一失败原因或真实身体改善。旧身体AB失败与本次静态验证场景不同，不能直接把RMS跨场景差值当独立因果效应。

## 下一最小动作

当前不进入新身体试验。需检查损失在长neutral尾段和短恢复窗口的分配，预先登记单个分段权重候选、保持架构/输入/运动权重不变，使用新独立验证配置；本次12配置今后归开发历史。不得在原目录追加1000、挑500、改门、设runtime死区/强制归零或reset。该后续尚未运行。

所有必要失败权重/记录/数据保留，0无用临时碎片。当前主目标继续active，未完成身体刺激恢复或空间趋近。
