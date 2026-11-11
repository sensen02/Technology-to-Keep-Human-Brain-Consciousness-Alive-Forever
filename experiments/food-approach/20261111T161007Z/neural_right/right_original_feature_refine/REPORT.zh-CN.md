# RIGHT原合格轨迹全特征有限2000：30.632932°，LH滑移.503296仍严格FAIL

固定一次新候选已完整结束，未扩预算。baseGRU128严格冻结，从固定ef768全部语义矩阵（含LH分片head重组dense42）初始化，encoder+GRU+head全部8张量43242参数训练；同RIGHT_ONLY/bound3/alpha/yaw/.06，仅原qualifiedteacher一条2880轨迹的post960动作MSE，无Dagger/CORRECTIVE loss。efactual流只用于初始same-state位对位预检。

最终模型：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/right_original_feature_refine/TRAIN/checkpoint_002000.pt`。

SHA256：`b4ed3e0ce3b6309a7e69c2379a1959c873c072fb0ba7c4124cb7db9a39b275e2`。

父ef768 SHA256：`ef768c788c94aeecbb49f0bd1ccb5c424333af36ec514941b72e49ab0d4917ff`，父4000仅血缘，不续旧Adam。freshALL8 Adam.0003/seed20261111/float64/CPU1，完整2880 BPTT、prime960无loss但不断hidden、不detach；冻结base输出/当步history缓存，共享同对象encoder/GRU/head的原TorchscriptScan。预检teacher和efactual两流初始means/basehidden/residualhidden bitexact、实际production/scan全部梯度/unitAdam/frozenbase/neutral全部通过≤1e-12。

训练固定2000更新，wall1160.59s，post动作MSE .000193249200→.000102549009。仅保存1000/2000完整性状态，物理仅final2000；保存时live模型/ALL8Adam/RNG/curve严格一致，load构造后恢复liveRNG。训练退出后的两个新进程保存审核和preCore准备均exit0，实际生产mean/全hidden重放与scan/neutral/base/源数据身份通过。没有加载或再生成oracle标签。

真实final RIGHT完整1s稳定+4.8s原straightteacher prime+9.6s ANN，新right命令1s smoothstep至-1，不reset；全部15400个1ms端点、2880个实际神经决策/192hidden/action/geometry/handover，1,232,000 native12.5us步（ANN768,000）已保存。原完整71opaque/49qpos48qvel/42有限PD、全部compiled/body/gravity/friction参数不变，无adhesion/extra、实际运行rootwrites0。

| 指标 | actual |
|---|---:|
| signedRIGHT yaw | -30.632932°，≥30°通过 |
| max slip | .5032961137641687，LH，严格FAIL |
| LF/LH/LM slip | .3847635749754991 / .5032961137641687 / .46311340972614856 |
| RF/RH/RM slip | .31941172639410687 / .45546755050513316 / .42289799846088294 |
| support mean/W | 1.0000690041563827，通过 |
| COMmin原生长度，1ms | 1.2552305090810747，通过 |
| world-X signed速度 | .08178757572713395，通过 |
| native ANN nonfoot peak | 0，严格通过 |
| 六腿LF/LH/LM/RF/RH/RM完整周期 | 1/2/2/2/2/1，通过 |
| 整体 | 原slip≤.5未过，严格FAIL |

完整结果：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/right_original_feature_refine/EVAL_right/RESULT.json`。

独立新进程实际审核：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/right_original_feature_refine/VERIFY_right.json`，147项完整性全部通过，行为FAIL保留。审核重放所有2880实际means/192hidden/action，校验源/input/artifact SHA、全端点指标/周期/来源/动作持有/感觉历史/连续交接、原生NF/extra计数、完整body/compiled与有限PD、无根状态写入。不是将训练MSE当行为结果。

因RIGHT未通过，依事前规则没有此独立候选neutral实际回合；离线2880 neutral输出/basehidden严格原直行仍通过，但不冒称跑过新neutral物理。bash42训练、43两个save审核+prepare、44实际RIGHT均exit0并已collect；实际独立审核前台exit0。没有追加2000、改LR或再开训练family。

父随后明确仅用于用户可视研究复核而posthoc选择本候选作combined_turning的RIGHT分支：.503296比ef .509829更低且其他门/30°/all6/NF0保留。`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/combined_turning/AMENDMENT_04_POSTHOC_VISUAL_RIGHT_COMPONENT.json`在任何组合assembly前明确记录；它不是原qualified-only替换规则的PASS，不是预登记性能验证，不把本严格FAIL改为PASS。组合模型将另外新SHA/三状态256/同SHA三实际与独立审核，再由父渲染研究样本；此候选没有单独新视频。

限定结论：本固定特征精炼改善qualifiedteacher拟合及本次实际slip，但仍未过原严格门，不能称完整大角度RIGHT工程验收通过。只属替代ANN/原声明派生完整身体/有限PD、单qualifiedtrajectory/标称9.6s；非完整nativeD1—D8、原生slip重新认证、connectome/SI/biology/food/连续命令域稳健性证明。原源/权重/失败/视频及原直行证据不改，tracker父负责。
