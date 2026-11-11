# RIGHT_ONLY解剖LH455承重行候选：30.631151°，LH滑移仍FAIL

本阶段固定预算全部结束，所有job已collect，无运行任务。实际RIGHT9.6秒右转30.6311514969°，六腿周期/严格原生非足0/支撑/COM/原world-X推进通过，但LH滑移.5098287866>.5，整体FAIL。比父uniform head候选.5106659712仅降低.0008371846，不能说已修复。没有追加训练、换LR、择优换档或重跑物理。

最终模型：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/left_hind_stance_refit/TRAIN/checkpoint_004000.pt`。

SHA256：`ef768c788c94aeecbb49f0bd1ccb5c424333af36ec514941b72e49ab0d4917ff`。

父模型：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/head_refit/TRAIN/checkpoint_004000.pt`，SHA256 `6407d6dbb85e1a7f40bf915e959433aa8c0896752924f06fab302e407b8f59b5`。名称left_hind只指LH解剖腿，不是左转命令；部署仍RIGHT_ONLY yaw/.06、forward0/turn[-1,0]、relu(-turn)*3*tanh(head)+frozenbase latent，最终actiontanh。与旧yaw/.008分开。

## 实施与实际预检

先预登记/code-only，父完成左教师物理并明确释放资源后，才开始本次诊断/预检/训练/两回合真实物理。

冻结validated base、encoder、GRU以及所有非LH head行。把原42x64 affine head参数分片为不可优化完整buffer和LH rows7:14的7x64+7可学习参数，只455参数requires_grad；forward每次组装完全相同形状矩阵后一次F.linear42，函数和部署输入不变。没有梯度hook或Adam尾部冻结行漂移。严格新格式独立loader验证冻结buffer/base/encoder/GRU与父模型完全相同，Adam仅两个LH形状状态。

训练用同一条完整真实RIGHT教师轨迹及实际连续hidden缓存，prime960保留但不计loss。仅LH7动作的post960加权MSE：实际teacher动作前LH足载≥.1W时权重4，否则1，以7×sum(weights)归一化。载荷只进入训练loss，不进入部署感觉、gate或神经状态。实际映射从冻结`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/sources/repair_core.py`的LEG_ORDER及原XML actuator joint顺序核验，LH contact-index1/head rows7..13。动作前载荷取TRACE index999+5*k，首post960为5799，与全部2880个SAMPLES.actual_time精确匹配，实测时间差0，未用动作后1ms/5ms载荷。

父6407模型同教师state下LH plain MSE .0001331440797，stance MSE .00008790607991、nonstance .0004791536459；stance占88.4375%行、占LH动作误差58.3893%。初始weightedLH MSE .0001002894503，实际LH梯度norm .01811996872/absmax .003107809282，有限且非零。

完整生产sequence/cached means最大差1.333e-15，全部base/residualhistory严格一致；LH梯度差≤1.421e-17，临时unit Adam参数差≤1.111e-16，均满足1e-12。原始完整模型输出初始严格父模型，unit update后相同输入的全部非LH生产输出仍严格父模型，所有冻结权重不变/无梯度，中性2880严格原直行。证据：

- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/left_hind_stance_refit/PREFLIGHT.json`
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/left_hind_stance_refit/PARENT_LH_DIAGNOSTIC.json`

## 固定训练与保存审核

freshAdam.001、seed20261111、float64/CPU1、固定4000更新，实际10.13秒（优化约8.72秒）。最终weightedLH MSE .0001006479322，比起点高.000000358482，不称拟合改善；final plainLH .0001395020094、uniformall42 .0001932491996。严格选择固定final4000，2000只是完整性存档，不跑物理。

两个保存档live模型/Adam/RNG/curve/frozen状态严格一致，临时strictload后恢复live RNG。训练结束后新进程独立各16项审核全通过：实际生产means/basehistory/residualhistory逐位重放，缓存/weights SHA、实际preaction mask、LH455/Adam2状态形状计数、非LH same-input输出严格父模型、frozen全状态及neutral2880严格基线。

- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/left_hind_stance_refit/TRAIN/VERIFY_002000.json`
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/left_hind_stance_refit/TRAIN/VERIFY_004000.json`

## 真实物理与独立审核

最终模型严格顺序neutral→RIGHT，每项完整1s稳定+4.8s原straightteacher prime+9.6s ANN，连续handover不reset，RIGHT新命令1s smoothstep至-1。每项完整15400个1ms端点、2880个5ms神经决策、192维hidden/实际动作/geometry/handover、1,232,000原生12.5us积分步（ANN768,000）。全71opaque mesh/49qpos48qvel/42有限PD、全部compiled数组/执行器参数与原基线严格一致，无adhesion/extra力/运行根状态写入。原五工程门与1ms slip口径未改，strictNF逐原生步轻量峰值记录。

| 指标 | 父head6407 RIGHT | LH455 ef768 RIGHT | 判定 |
|---|---:|---:|---|
| 实际yaw度 | -30.6899178407 | -30.6311514969 | ≥30°通过 |
| LF slip | .4189846709 | .4098881260 | 通过 |
| LH slip | .5106659712 | .5098287866 | 仍FAIL |
| LM slip | .4752110321 | .4740876590 | 通过 |
| RF slip | .3272844806 | .3274102588 | 通过 |
| RH slip | .4608236058 | .4664102720 | 通过 |
| RM slip | .4546797106 | .4486675386 | 通过 |
| 足载均值/W | 1.0000690860 | 1.0000645498 | 通过 |
| COM最低，1ms | 1.2552403408 | 1.2551137298 | 通过 |
| 原world-X signed速度 | .08186511465 | .08179755890 | 通过 |
| ANN原生NF峰值 | 0 | 0 | 严格0通过 |
| 六腿周期LF/LH/LM/RF/RH/RM | 1/2/2/2/2/1 | 1/2/2/2/2/1 | 全六腿通过 |
| 整体 | FAIL | FAIL | LHslip>.5 |

NEUTRAL五门/all6/nativeNF0通过，slip .49278883265360646、yaw -.3825987542°，所有共享完整物理数组与原直行基线严格相同。非LH head参数严格保留只保证相同输入/hidden下输出相同，不意味着RIGHT闭环物理状态改变后其他腿实际动作时间序列不变。

新进程独立审核零积分/零优化：

- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/left_hind_stance_refit/VERIFY_neutral.json`：173项完整性全部通过，行为PASS。
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/left_hind_stance_refit/VERIFY_right.json`：149项完整性全部通过，行为LHslipFAIL。
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/left_hind_stance_refit/EVAL_right/RESULT.json`：同次全部实际物理指标和artifact路径/SHA。

审核包括身份/source/data/artifact SHA、所有实际2880 neural outputs/192state、实际动作持有/来源/感觉上一动作、全物理/geometry时钟/连续交接、strictloader冻结行约束、实际native完整计数/NF/extra、原全compiled/有限PD不变以及无rootwrites源码路径。

## 收尾与下一动作

bash20预检+固定训练exit0、bash21两保存审核+prepare exit0、bash22完整neutral exit0并独立审核exit0、bash23完整RIGHT exit0并独立审核exit0，全部job已collect，无运行任务、无无用临时碎片。没有改旧candidate源码/权重/失败；tracker父负责。

本候选没有修复LH滑移，也没有比初始weighted loss更好，不据此认定承重动作误差为根因。固定预算终止，不立即追加同Adam/BPTT。父可另立有收敛依据的readout拟合或以真实slip为目标的小维神经参数物理校准，仍须先登记与同窗30°/全部原门；本子未执行额外候选。

仅为已授权替代ANN/原声明派生完整身体/有限PD、单真实教师轨迹与标称初态9.6s工程回合；不是完整原生D1—D8/原生slip独立认证、原连接组/SI/生物有效性或任意扰动稳健性。此模型没有同次视频交付；父已制作的f25b模型视频属于旧候选，不冒充此模型。
