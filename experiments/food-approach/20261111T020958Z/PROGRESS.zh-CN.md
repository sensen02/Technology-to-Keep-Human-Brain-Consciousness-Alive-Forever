# 续轮3非线性正向神经读出实验（进行中）

本目录ROOTPREREG在任何结果前登记。父模型为既有head2000 eeced...，旧实际中性/右工程PASS、左RHslip.514841585FAIL。旧head固定续10000未改善left.515626812，停止在同一线性头上继续加步数。

新增MLP306→64tanh→42zerooutput，当前允许物理112＋basehidden128＋residualhidden64＋command2，relu(actualturn)*.10*tanh追加latent；freeze父全部197608参数和两组记忆，只有新22378ANN参数训练。负/零命令新支路恒0，但仍要求实际全轨迹复核。没有时钟/步态phase/食品坐标/来源输入、脚本步态、根轨迹或额外物理支持。

独立只读诊断确认旧左RH vt约比teacher高.078–.081，进展分母几乎相同，承重>=.1W贡献约.961分子且整窗存在；旧线性head不饱和，不能把本试验称为加幅修复或唯一根因证明。诊断及所有输入来源已核验，0优化/积分。

训练childbash3固定5000exit0收集，更新62.58s/总70.50s。left教师actionMSE8.720976e-5→3.811663e-5，原合格left_full实际2880序列/960prime不计loss，freshAdam.001，新head only。2500/5000各19项独立保存核验PASS，全部冻结父权重/实际Adam4状态/counters/RNG/curve0..N/negativezero2880输出与192hidden精确；livepresavefullstate匹配并恢复构造后liveRNG。只实际选最终5000。

最终checkpoint：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T020958Z/train/TRAIN/checkpoint_005000.pt`
SHA：`8e341270d82d923dcf19ec5b7286fc5797806688b3f27567f75ea3336f7216bd`

评估器strong43checksPASS，实际最终5000 neutral/right各2880生产尺寸mean/actions/192hidden与合格父精确；原score/command AST、篡改memory拒绝、全部28 source_paths存在/hash检查在Core之前完成。首次准备结果JSON的np.bool_错误在无物理阶段修正并重跑全部检查，不以只改哈希替代运行。所有captured源码冻结。

parentOWN新runtimebash7正在运行同一检查点中性/L/R三回合，原full1ssettle+4.8实际teacherprime+9.6ANN，native12.5us/42finitePD/71opaque/无adhesion/外力/根写入；原全部五门/all6/NF0/yaw.01不变，立即保存HANDOVER和异常部分记录。中性已actual+auditPASS slip.49278883265，左右待完成；不能提前声称左过门。

下一项steering_holdouts/仅CODEONLY草案，必须现同SHA三条件全PASS及审计通过才注册实际4个命令变化+2个20ms有限motorpulse初态案例。尚未执行、不计食品资格。神经减速/实际食品行为训练/留出布局/切断重放和同次正式native证据仍是待完成目标。
