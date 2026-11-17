# 动作MSE与增量目标累积：仅可检验方案提案，未实施候选

本文件只做源码机制和可验证性分析，没有新模型／训练／预算，没有NumPy／Torch数组、权重或物理执行。230738候选与更早科学源、数据、mask、失败及运行中214身体实验均保持冻结。父已报告212离线600与保存审核完成，214首walk仍slip1.09245失败，其余结果尚待实际汇总；这些信息不证明下述机制是唯一原因。

## 1. 动作即时误差如何进入有记忆的执行器

原 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/sources/repair_core.py:250-252` 明确normalized action∈[-1,1]42先乘MAX_RATE=2rad/s再限幅。`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/physics.py:139-164` 每1ms调用actuator.update一次、随后80个12.5µs物理步；一个200Hz神经动作保持5次1ms。target不是姿态或速度命令本身，而是持续积分寄存器。

`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/sources/incremental_actuator.py:170-193` 的真实更新是：

```text
r = clip(2 * clip(action,-1,1), -MAX_RATE, MAX_RATE)
r_limited = clip(r, -actuator.max_rate_rad_s, actuator.max_rate_rad_s)
target_next = clip(target + .001*r_limited, target_bounds[:,0], target_bounds[:,1])
```

真实MAX_RATE和max_rate须从实际IDENTITY核对，不能凭默认值放宽；当前原生接口默认2。按源代码逐1ms五次更新保留，不为方便换成一次.005大步，即使单调常动作某些情况下数学等价，也要验证舍入／边界行为。rate先裁再积分、角度随后裁，达到角界不继续累积windup。零动作只保持已有target，不把target归零。

无触边时，两个序列的目标差近似 `ΔT(k)=ΔT(0)+.01*Σ(action_pred−action_source)`（每5ms的2*.005）。均值为零但有时序相关的误差仍可能积累，有符号小偏差则可持续漂移。举例仅作量纲说明：若一个通道normalized action平均偏差+.01、持续9.6s且从未裁剪，目标差约+.192rad；不是本次实测值，也不暗示全部动作差有该偏置。独立白噪声、偏差和相位误差不同，不能只用全42平均MSE判断目标漂移。

即使每步MSE降低，误差符号、跨时间相关、支撑腿关键通道、触边／反向时刻或失载相位不同，最终target与接触运动仍可能更差。全42均值也可能掩盖某通道长期偏差；行走slip对承载时接触点速度和净推进分母都敏感。但没有保存数据分析前，不把此解释当已证实根因，不能反推所有闭环失败都是积分造成。

有限PD不是由loss直接命令的额外控制器：`incremental_actuator.py:195-206` 每物理子步用当前实际q／qvel算 `clip(kp*(target−q)−kd*qvel,±torque_limit)`。目标误差会改变力矩，但实际惯性／接触／支撑和饱和决定身体运动；固定观察目标展开不能替代物理验证。

## 2. 确认当前target感觉及旧神经prior

`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/sources/proprioception.py:29-43`：halfspan=max(native joint range跨度/2,.1)，q0取实际keyframe关节参考；42:84=(target−q)/halfspan，126:168=(target−q0)/halfspan。先整体clip[-10,10]，再astype(float32)。0:42是当前qoffset、84:126为qvel/10；没有把126:168误称effort。

`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/models.py:50,113-116,137-143` 的physicalmask使old256递归编码不直接读取两类target和previousaction；但整个oldmean显式包含 `−softplus(raw_gain)*obs[...,126:168]`。因此可写成冻结原神经部分与有符号target-position prior之和，不能称wholemean对target独立。230738 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T230738Z/local_mean_corrective_calibration/model.py:82-102` 新encoder额外读取同次直接oldmean42，它也能因果感知该targetprior；不是教师q_des输入。

## 3. 最小可检验顺序：先验证执行器代数，再讨论训练

第一步可在未来单独授权的saved-array诊断中，不修改X原数据，使用**真实应用动作**和源target初值，重放纯target积分：

- 200Hz源X在动作计算前。物理ms=5800时preaction target=trace.target[5799]；该次动作的五个更新与post targets对应trace.target[5800:5805]，末slot trace[5804]即5.805s。不能拿第5800行当初值，造成一毫秒泄露／错位。
- 原teacher2880序列的warm索引0:960实际time1.0..5.795，ANN式3080源前200settle没有motorstep，warm200:1160；自主后缀两者都从物理5.8s开始，但文件行号不同，必须用保存physical_ms/time核对。
- 对每条源先teacher-forced实际应用action重建全部1ms target，要求与保存target最大误差≤1e−12（并逐项记录；浮点若不满足先诊断实现，不开始loss）。检查first／last、边界角、速率cap／角度clip mask与dtype，不以近似大步冒精确。
- 用实际IDENTITY的target_bounds/max_rate/kp/kd及正确42关节qpos地址／q0／halfspan重建两个target感觉片段。首选原完整模型映射或被保存并hash绑定的映射，不能假设任意joint数组顺序；源XML42actuator映射可作来源。teacher-forced真实target经过clip→float32→float64后的片段应与原X字段逐位一致。若只拿float32 X反推q/target，会因舍入／感知饱和丢信息，不能当真实rad目标。

这些验证可以证明训练参考的积分时钟与归一化正确，但不证明学生预测动作施加到真实身体会产生同一q／载荷。任何新的训练loss都须先有这个可重放来源契约。

## 4. 条件性目标反馈展开的定义

若父在214实际结果后选择这条路线，可另立一个明确的离线conditional rollout：起点使用源真实target及原old256/new96 warm状态；保留源qpos/qvel/loads／clearance／身体倾角／速度等真实观察记录，只在每个5ms动作计算前，以当前**预测target寄存器**替换内部派生输入的42:84与126:168，按同一halfspan/q0/clip/float32流程编码；不改磁盘原X。

网络输出预测action后，按真实每1ms五次增量规则更新预测target；下一决策再读取这个预测target。它没有给执行器添加新算法，原执行器本来就是Other工程环；训练展开只让神经网络面对自身累计命令对当前寄存器反馈的影响。未来部署仍运行既有NN→原增量PD，不把teacherclock／idealroot／私有q_des塞actor。

有两种不同的固定观察控制需分开说明：

1. 为诊断targetprior的纯代数效应，可冻结保存的oldhidden／physical history，只用显式旧targetprior随预测target改变。这个结果只是一种条件性观察对照，不能叫完整生产网络递归展开。
2. 为训练与生产一致，应在固定源实际物理通道之上，按生产公式递归推进old256和new96；old256的mask不直接读target，因此在给定同一外生physical历史／task和初态时其路径理论上不受两个替换片段影响，但仍要数值比较。oldmean、encoded和new96则可能因targetprior变化而改变。不能把不同控制条件混成同一种闭环。

两者都没有重新生成预测动作导致的新qpos／qvel／足地载荷；原身体感觉仍是外生源历史。因此它们是条件性神经—执行器寄存器展开，不是真正动力学闭环、不是物理反事实成功，也不能用预测target误差下降替代新的完整身体slip／支撑／周期验收。

## 5. 哪些target可以作真监督

原8条物理合格源的Y是实际应用动作，若上述积分重放契约成立，其真实Trace.target可以监督预测target路径。221508 old_walk_control也实际应用direct789f动作，故这第9条可同样提供真实累计target监督。原8不因B后来保守mask被追改。

三条assisted真实动作是.75*teacher+.25*student，其Trace.target积分了**混合应用动作**，而训练Y是unmixed full teacher_action。若用unmixed预测teacher动作积分，却强迫其target匹配混合Trace.target，会给同一个样本不一致目标。不能这样做。

对这3条，可保留现有label_qualified上的即时teacher action监督；若未来要target监督，至少区分：

- 只做one-step局部监督：从真实当前target出发，积分该时刻合格teacher action五次得到的参考target，是局部未实现的建议，需要如实标识，不是实际Trace.target。
- 用完整teacher action序列积分出虚拟target参考是另一个更强的条件性标签；teacher可能在混合执行真实身体状态上计算，其沿时序并不等于纯teacher真实闭环轨迹，不可称已实现教师target。invalid动作不能当合格监督，不能跨无效段硬造连续teacher参考。

因此最保守提案可只对9条真实应用目标来源增加累计target监督，另外3条保持masked即时action loss；每条如何等权、masked gap如何处理、权重与训练预算都留待新的预登记，不在本文件填数。

所有序列的预测target／hidden历史即使invalid supervised mask为0也仍应按合法学生动作推进；不能删无效行、resettarget或把缺监督目标填0。失效样本仍须原样保存、有效mask不能冒物理PASS。

## 6. 可微实现需要哪些明确边界

当前230738模型cached_features在torch.no_grad内运行old.step并拼features；旧训练直接缓存frozen oldmean／oldhidden。因此若目标输入改为依赖预测target，直接复用detached缓存会截断target→oldprior→newbranch→action→下一target的梯度。这个限制不能用“旧权重冻结”混过去：**冻结权重不等于detach输入**。

训练专用实现可保留旧权重requires_grad=False，但在需要路径上允许输入梯度；也可在已证明oldhidden对target替换独立的条件下，复用oldhidden神经部分并明确写出同一冻结 `−gain*target_offset` prior，和生产公式核对。它必须数值复现oldstep，不是用新非神经控制器替代NN。理想教师q_des仍只作标签求解，不进该公式。

float32感知转换是实际量化，不能悄悄去掉。PyTorch浮点dtype cast的autograd能传梯度，不需要额外straight-through估计器；但这不意味着真实舍入映射在严格实数意义可微。数值资格需把生产值一致性与所用autograd近似分别说明，至少teacher-forced输入／目标重放、中心差分适用范围（避开clip边界和量化台阶）、完整序列梯度及保存重放要分层核验。

clamp在饱和区的梯度平坦是模型事实，不能为得到漂亮loss而事后删触cap样本。角度／速率cap占比与连续饱和时长应单列；预先定义是否保留饱和段监督及不可导边界处理，不以工程可微近似替换真实执行规则后仍声称同一口径。

## 7. 先做什么诊断才决定路线

可先从已保存真数据检验：逐通道有符号action误差、按时间的累计误差与真实target差、目标触边／反向、支撑时载荷与足速度的对应；区分首步突变、持续偏差和相位错误。teacher-forced实际action积分必须先复现保存target。然后才比较固定观察下“完全真实target强制输入”与“预测target反馈输入”的动作／目标误差，且两者都不推测物理成功。

214全四项实际结果与这些诊断证据出来后，由父决定是否新预登记累计target loss、继续当前预算路线或其他神经任务机制；本文件不授权或实现新候选，不加层数／参数／epochs，不改任何源或mask。所有旧失败、原789f行走通过与未来realDN2→decoder2/full432接入要求保持。
