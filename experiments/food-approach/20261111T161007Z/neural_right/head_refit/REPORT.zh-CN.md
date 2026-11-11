# RIGHT uniform head-only固定4000：30.689918°，LH滑移仍FAIL

本阶段全部已授权训练/物理/独立审核结束，无运行任务。冻结base+encoder+GRU，从原RIGHT2000模型只训练2730个head参数、原uniform post960动作MSE、freshAdam.001/fixed4000。训练及两保存完整性通过，中性严格原基线；最终RIGHT转30.6899178407°，最大滑移从.5224375036降至.5106659712，但仍超过.5，不能判通过。没有追加预算或择优改档。

最终模型：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/head_refit/TRAIN/checkpoint_004000.pt`，SHA256 `6407d6dbb85e1a7f40bf915e959433aa8c0896752924f06fab302e407b8f59b5`。2000完整性存档SHA256 `312df2d2d13b6aefd0bc4aaf846d4cc7bc6935291382e28b4d7c9015da1c051a`，未做物理。父模型为`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/TRAIN/checkpoint_002000.pt`，SHA256 `f25b920b1d9bc74a2ff3092dd926908d728b1b0ee49d211d4295e59f630aee15`。

先独立PREREG/code-only，父完成其他物理并明确释放资源后执行。没有改变新RIGHT_ONLY yaw/.06命令域、gate relu(-turn)、latent bound3、网络结构/身体/执行器/时间窗口/门槛，明确区别旧yaw/.008。训练full2880真实教师连续history缓存，只对post960的1920行动作计原均匀MSE，不detach或重新生成hidden，不做全网BPTT、不作stance weighting。

生产sequence/cache最大means差1.777e-15，base/residualhidden严格一致；head梯度差2.365e-18、临时unit Adam head权重差1.685e-14，均≤1e-12。全部冻结权重严格相同/无梯度，初始完整权重和实际means/hidden严格父模型，中性2880严格原基线。证据`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/head_refit/PREFLIGHT.json`。

固定4000更新训练wall11.09秒（优化约9.92秒），MSE .0002159072721→.0001921895446，约为父模型.89015倍。freshAdam只含两个head状态；没有继承旧Adam。1000之外无新预算：本候选只按预登记save2000/4000，每次live模型/Adam/RNG/curve严格保存，临时strictload后恢复live RNG。独立新进程两个保存档各13项全通过，实际生产means/basehidden/residualhidden逐位重放、冻结权重严格父模型，缓存SHA绑定、neutral2880严格基线。证据：

- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/head_refit/TRAIN/VERIFY_002000.json`
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/head_refit/TRAIN/VERIFY_004000.json`

每个物理回合完整1s稳定+4.8s原straightteacher prime+9.6s ANN，1s smoothstep新RIGHT命令，不reset。49qpos/48qvel/全71opaque/42有限PD/原生12.5us/无附着/额外力0，全部compiled数组/执行器参数与原基线严格一致；连续192hidden、几何、实际动作/trace/handover全部保存。每回合15400端点、2880决策、1,232,000原生步，ANN768,000步。源/输入/artifact SHA全部验证。

| 指标 | 父RIGHT2000 | 本head4000 RIGHT | 结论 |
|---|---:|---:|---|
| yaw度 | -30.1385330474 | -30.6899178407 | 右转≥30°通过 |
| LF slip | .4337565017 | .4189846709 | 通过 |
| LH slip | .5224375036 | .5106659712 | 仍FAIL |
| LM slip | .5031381783 | .4752110321 | 本候选通过 |
| RF slip | .3297169928 | .3272844806 | 通过 |
| RH slip | .4925245749 | .4608236058 | 通过 |
| RM slip | .4753323670 | .4546797106 | 通过 |
| support均值/W | 1.0000662863 | 1.0000690860 | 通过 |
| COMmin，1ms | 1.2554756289 | 1.2552403408 | 通过 |
| 原world-X signed速度 | .08083452406 | .08186511465 | 通过 |
| native NF peak | 0 | 0 | 严格0通过 |
| 六腿LF/LH/LM/RF/RH/RM周期 | 1/2/2/2/2/1 | 1/2/2/2/2/1 | 全六腿通过 |
| 整体 | FAIL | FAIL | 不放宽slip≤.5 |

中性：slip .49278883265360646、yaw -.3825987542°、五门/全六腿/native NF0通过。全部共享15400物理数组和2880神经means/action/basehidden严格等于原验收直行基线，另有全部192hidden生产重放验证。

独立新进程保存数组审核：

- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/head_refit/VERIFY_neutral.json`：168项完整性全通过，行为PASS。
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/head_refit/VERIFY_right.json`：144项完整性全通过，行为LH滑移FAIL。
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/head_refit/EVAL_right/RESULT.json`：完整实际逐项结果与artifacts。

运行记录：首次bash14在初始化前因父重跑旧teacher VERIFY.json加入一个cycle检查造成旧data SHA漂移而exit1，0训练/0物理、未建TRAIN。已只读证实删除新增字段并indent2+newline可精确恢复原SHA，父保存扩展审核为单独补充后恢复原字节；未重锚checkpoint或旧hash。bash15同科学参数重试exit0，bash16独立两保存审核/prepare exit0，bash17 neutral exit0及独立审核exit0，bash18 RIGHT exit0及独立审核exit0。全部后台job已collect，无运行任务。失败日志保留在`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/head_refit/TRAIN.log`，实际训练在同目录`TRAIN_RETRY.log`。

本阶段证明该限定候选降低teacher MSE及本次真实slip、LM由失败转通过；没有证明LH根因或保证继续拟合必然达标。仅标称初态/单条教师训练/9.6s、替代ANN/原声明派生完整身体/有限PD的工程回合；不是完整原生D1—D8、原生滑移重新认证、原连接组/SI/生物有效性/任意扰动结论。没有为本候选制作视频；父先前视频属于父模型f25b...，不能替代此6407...模型的同次视频。

后续由父提出新LH承重行候选，先补已有模型teacher-state LH误差/梯度诊断并单独预登记code-only，不在此阶段继续训练或物理。
