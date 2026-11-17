# 第二版身体pilot：只更换实际聚合校准final400权重

新目录独立于已归档的首批六身体回合。父实际反馈域诊断表明首AN冷／暖不是主要偏转解释，聚合校准训练在新timestamp完成后，只有实际final400及其保存／独立审核证据可由父绑定。旧六回合滑移和恢复失败不改；训练拟合改善不是新物理通过。

本子仅CODE_ONLY：38项标准库检查实际全部通过exit0，0Torch／NumPy／数组／模型forward／Core／物理。所有执行源码自READY后停止修改，只有本说明为纯文档未纳freeze。

## 父实际命令

1. 绑定父已完成的actualfinal400：

`python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T000915Z/body_pilot_v2/prepare.py --bind-completed-calibration --brain-sha256 415304b7027dda17f359350052934670c1aa2bd44a0025b1dd0089d549997b27`

2. 父实际0Core预检：

`env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T000915Z/body_pilot_v2/preflight.py --parent-resource-released`

3. 父唯一托管重任务：

`env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T000915Z/body_pilot_v2/orchestrate.py --parent-resource-released > /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T000915Z/body_pilot_v2/PIPELINE.log 2>&1`

第1—3步本子未执行，不生成假准入或预检。SHA由父实际训练结果提供，通过参数和INPUT_BINDINGS在运行时绑定，没有把未知值硬写进runtime源码。prepare核新common中FORMAT SOURCE_NECK_FEEDBACK_AGGREGATION_V2、实际七阶段exit0、存储与独立sourcequeue／evaluation审核，final400而不是旧200。没有新增离线allfunctionalPASS门。

## 最小复用和身份范围

runner、audit、summary、orchestrate通过一个AST层复用冻结旧directloop，所有物理／时序数值常量逐项与旧源码一致；BrainNeckLoop每方法codeobject保持同一，完整globals ROOT/CAL/newBRIDGE/newSHA绑定到新timestamp，真实端口、motor和物理源仍指原不可变文件。

唯一数值比较变更是预检bp.step200→400、准入brain_final_step400；旧settle200记录计数、3080／272／1160handover等全部保持。路径TRAIN/VERIFY_000400.json与新checkpoint_000400.pt匹配。新model.py顶层需要common，loader将newcommon与newmodel都临时alias绑定，随后motor alias，finally恢复common/model/sys.path，不依赖污染的basename缓存。

同六条件P_empty/Lpulse/Rpulse/L_DNcut/L_ANcut/L_cuezero；同1＋4.8＋9.6秒、15400身体行／3080神经行、272状态，heldAN1/DN2另记，1msqueue／2ms发布／5ms消费。warm AN从实际t0持续，脑16保持零／DNzero预热，teacher960次motorcommand0与shadowdecoder分列；handover不清queue或状态。切断ANN1.6只静默对应heldport及丢在途，不runtime归零turn或减bias，所有物理门和失败分列规则不变。

原模型的无目标偏转、AN反馈行为与全部slipFAIL都是历史真实结果。新模型即使离线silentdecoder接近0，也必须在此六完整闭环中独立验证，不能提前说脑经真实颈部已恢复正常行走或启动／停止控制。真实AN工程编码、native细胞功能、电极及意识结论仍未建立。
