# 独立RIGHT_ONLY大角度ANN：30.1385°已实现，滑移未通过

本次S4/S5固定预算任务已完成：新模型训练、最终模型两次完整真实物理回合及新进程独立审计均结束。RIGHT实际9.6秒右转30.1385330474°，六腿周期/严格原生非足0/支撑/COM/原world-X推进通过，但LH滑移0.5224375036和LM滑移0.5031381783超过0.5，因此该模型整体FAIL，未覆盖原通过直行或旧小转模型。中性回合与原直行基线全部共享保存数组严格相同。未追加训练或物理搜索；视频按委托由父另行处理。

## 模型、训练与身份

- 最终模型：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/TRAIN/checkpoint_002000.pt`。
- SHA256：`f25b920b1d9bc74a2ff3092dd926908d728b1b0ee49d211d4295e59f630aee15`。
- 1000档完整性存档：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/TRAIN/checkpoint_001000.pt`；SHA256 `46d1fea46d2aa557a85a2c90782e4de8605ed5e5da2cc4b3d1cdcff24caa7cf9`，未做物理。
- 冻结原验收walking GRU128；其checkpoint SHA256为`246f9fab7d21e2a1332a745fafd4a343b80ad5a70d2cafeeb900e876f7933809`。实际感觉112+basehidden128+command2，encoder64/tanh、slow GRU64，alpha=1-exp(-.005/.1)，head42；base latent加`relu(-turn)*3*tanh(head)`，最终动作tanh。域为forward=0、turn∈[-1,0]，新yaw/.06，明确区别旧yaw/.008。
- encoder/GRU迁移自`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/train_two_only_control/TRAIN/checkpoint_001500.pt`，固定SHA256 `c6b395f29676ef933e2f55ca5a92b7c5b55183a2d432226487b39e916b51acdb`，并验证其payload source/data。head全部zero；没有载入旧Adam。
- 单条完整真实RIGHT教师轨迹2880行，前960行连续prime但不计loss，其后1920行动作MSE。冻结base输出/历史缓存，同对象参数scripted scan，完整连续BPTT，无detach、无clock/phase/food坐标输入。
- fresh Adam .001、seed20261111、float64、固定2000更新，实际1159.68秒。初始MSE .0953057351，最终生产sequence重放MSE .00021590727210239838。声明的gradient clip1在全部更新中未激活，最大梯度范数.5704306331。
- 所有旧文件保持；新命令域、新界、新训练任务与迁移组合声明，不作单变量因果归因。严格新格式拒绝旧模型。

## 正确性与保存审核

`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/PREFLIGHT.json`全通过：非zero临时head下完整production step/sequence与cached/scan输出、全部hidden、8组adapter梯度和临时unit Adam参数/状态差均≤1e-12；最大前向/hidden差1.555e-15，梯度差1.076e-16，unit参数差1.381e-15。base无梯度/权重不变，中性2880行means及basehidden严格基线，初始RIGHT zerohead严格base。

每次保存时live模型/Adam/RNG/curve均与文件严格相同。训练退出后分别新进程strictload1000/2000，各11项审核全通过，重放生产means/basehistory/residualhistory逐位相同；scan最大差2.332e-15，两个中性重放都严格基线：

- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/TRAIN/VERIFY_001000.json`
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/TRAIN/VERIFY_002000.json`

保存时临时strictload构造模型会消耗live torch RNG，训练中没有修改源码补救。独立`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/TRAIN/RNG_AUDIT.json`实测1000 saved RNG经过实际loader后的状态精确等于2000 saved RNG、NumPy RNG不变，5项全通过。本次训练没有dropout/采样等随机更新，保存RNG可精确恢复；不声称任意随机训练过程的逐步续训RNG不变。

## 最终模型真实物理结果

严格顺序neutral→RIGHT，仅固定2000档，每回合1s稳定+4.8s原straightteacher prime+9.6s纯ANN。接管没有reset，RIGHT命令1s smoothstep至-1，教师仅prime调用960次、ANN段0次。每回合保存15400个1ms完整端点、2880个5ms实际决策、全部192hidden/mean/policy/实际动作、几何和handover。每回合1,232,000个原生12.5us积分步，ANN768,000步。轻量原生每步接触统计，避免巨量native文件。

| 同窗指标 | NEUTRAL | RIGHT | 门槛/结论 |
|---|---:|---:|---|
| 实际yaw度 | -0.3825987542 | -30.1385330474 | RIGHT右转≥30°通过 |
| 最大逐腿滑移 | .4927888327 | .5224375036 | RIGHT失败，阈值≤.5 |
| 足载均值/W | 1.0000510230 | 1.0000662863 | ≥.90，通过 |
| COM最低，1ms | 1.2561645313 | 1.2554756289 | ≥1.20，通过 |
| 原world-X有符号速度 | .07131859637 | .08083452406 | ≥.05，通过 |
| 原生ANN非足峰值 | 0 | 0 | 严格0，通过 |
| 原生ANN COM最低 | 1.2561640816 | 1.2554756038 | 补充原生记录 |
| 六腿周期LF/LH/LM/RF/RH/RM | 1/2/2/2/2/2 | 1/2/2/2/2/1 | 全六腿通过 |
| 最终工程结论 | PASS | FAIL | 不以角度通过替代全部门 |

RIGHT逐腿slip：LF .4337565017、LH .5224375036、LM .5031381783、RF .3297169928、RH .4925245749、RM .4753323670。body-heading投影速度.07220275778，仅补充，不替代原world-X门。RIGHT实际动作绝对最大.818709196、残差latent绝对最大2.330639338、端点ctrl/有限力矩上限最大.595708684；未见动作硬饱和，端点统计不证明所有原生力矩子步不饱和。

完整证据：

- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/EVAL_neutral/RESULT.json`
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/EVAL_right/RESULT.json`
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/VERIFY_neutral.json`：新进程156项完整性全通过。
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/VERIFY_right.json`：新进程132项完整性全通过，行为FAIL完整保留。

独立审核零积分/零优化：source/input/artifact SHA，实际2880 means/hidden/action逐位重放、所有实际动作5ms保持、感觉历史真实上一动作、连续handover、全71opaque mesh、49qpos/48qvel/42有限PD、原全部compiled数组/执行器参数不变、无附着、原生external0、实际tick及rollout源码无根状态写入/运行reset。NEUTRAL对原基线全部共享15400物理数组和2880神经数组严格一致；RIGHT完整5800端点预热也严格一致。保留原五工程门实现与1ms滑移口径；此次不是完整原生D1—D8/原生滑移独立认证。

## 已有数据诊断与下一候选依据

`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/ANALYSIS.json`、`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/SLIP_DIAGNOSTICS.json`和`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/STANCE_FIT_DIAGNOSTICS.json`仅分析已有轨迹和最终checkpoint，没有新物理/训练。

实际xy推进分母.08564560864/教师.08640761772=.9911812281，仅下降约.88%。LH载荷加权脚速是教师的1.08564倍，LM为1.11641倍；故滑移升高主要不能由推进分母下降解释。

三段各3.2s的局部slip诊断（不替代全窗验收）：

| 腿 | 0—3.2s | 3.2—6.4s | 6.4—9.6s |
|---|---:|---:|---:|
| LF | .35470 | .34971 | .59858 |
| LH | .48251 | .52311 | .54764 |
| LM | .50965 | .52136 | .44603 |
| RF | .37063 | .30895 | .30486 |
| RH | .48514 | .54843 | .41315 |
| RM | .51172 | .44265 | .47427 |

对全窗slip的可加贡献：LH .138844/.172262/.211332；LM .190955/.199625/.112559，三段求和与全窗一致。实际≥.1W承重段占LH滑移分子96.792%、LM92.821%；≥.2W段占LH85.196%、LM60.951%。承重段滑移值得针对，但不等于教师状态拟合误差一定集中这些腿。

教师同一感觉/连续hidden历史上的残差latent误差MSE .000296818739，最大绝对.4609134031；计算(mean-base)-(atanhY-base)精确消掉相同base。prime中存在边界动作±1，因此逆tanh只在排除prime后的1920行计算，无裁剪、所有后prime标签绝对值<1。初次只读诊断曾对被排除prime做逆tanh产生warning，已改为仅后prime并重检；数值结论未变，未重跑物理。

以实际teacher动作前足载≥.1W定义stance，并将trace row999+5*k与全部2880个教师输入时间逐一对齐：stance占70.616%关节样本，但只占41.454%动作平方误差；stance MSE .00012674325，非stance .00043019068。LH stance MSE .00009920153，LM .00011474564，均不高于LF .00020787062/RF .00019798466。LH stance占自身误差57.28%，LM41.46%。因此“LH/LM承重拟合误差特别高导致滑移”未获支持。

闭环晚段出现偏离：7—8s实际动作与同钟教师MSE .00496148、8—9s .00800308，而模型在原教师感觉/hidden历史上相应MSE仍约.00026415/.00022919。这是状态分布偏离的描述性证据，尚未隔离根因。

父提出的“冻结base/encoder/GRU，仅head在固定全轨迹hidden上做承重weight4/freshAdam.001/fixed4000”的后续候选可减少计算，并检验承重加权拟合是否有帮助；当前数据支持将其作为有预算的试验，不支持预先保证修复LH/LM或闭环偏离。本子任务没有执行该候选。父优先制作同次30°轨迹且明确FAIL的视频，再推进独立左教师/后续候选；本次不追加训练。

## 运行记录、限制与收尾

训练bash-7 exit0；保存审核bash-8前3项全过，prepare首次NumPy布尔JSON序列化exit1且Core=0，修正未捕获的准备脚本后单独prepare exit0；中性bash-9 exit0；RIGHT bash-10 exit0；独立RIGHT审核/只读分析bash-11 exit0；补充三段/stance诊断均exit0。全部后台job已collect；本子没有运行任务。一个CPU线程、训练RSS约489MiB；没有并发训练/物理进程，没有子代理。没有无用临时碎片；保留失败日志及必要模型/数组。tracker由父负责。

本结论仅为已授权替代ANN、原声明派生完整身体/有限PD、单轨迹训练和标称初态9.6s回合。不是原连接组、独立SI/生物有效性、任意时长/扰动稳健性，也没有独立视觉验收。旧通过模型和旧失败均保留。
