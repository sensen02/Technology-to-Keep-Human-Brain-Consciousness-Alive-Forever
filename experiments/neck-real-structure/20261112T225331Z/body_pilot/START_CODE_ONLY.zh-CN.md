# 人工脑—真实身份颈部—身体：固定六回合物理pilot

本子已完成CODE_ONLY，实际仅29项标准库检查通过exit0，未运行Torch／数组／模型forward／Core／物理或正式准入。源自READY后停止修改。父已完成的离线校准完整性、固定200最终权重与真实N2/M1来源绑定，是此有限工程闭环的前提，不要求离线开发反转功能全部PASS。

## 父准确命令

1. 实际绑定已完成的校准权重／源与冻结motor：

`python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T225331Z/body_pilot/prepare.py --bind-completed-calibration --brain-sha256 eed1ad3cc61a1b59086ff2324175f6bf3b5da5aa197165e6c4b009c17027c95e`

2. 父实际0Core预检：

`env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T225331Z/body_pilot/preflight.py --parent-resource-released`

3. 父唯一托管重任务，六物理／六独立审核／汇总串行：

`env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T225331Z/body_pilot/orchestrate.py --parent-resource-released > /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T225331Z/body_pilot/PIPELINE.log 2>&1`

本子不执行以上数值／物理命令，不输出假SHA／预检或结果。实际父prepare生成INPUT_BINDINGS/FROZEN前检查校准7stageexit0、存储／sourcequeue独立重放及actualevaluationaudit；函数开发失败不伪装成通过、不阻止这项明确的探索6回合。

## 六个条件与数据流

P_empty、P_left_pulse、P_right_pulse、P_left_DN_cut、P_left_AN_cut、P_left_cuezero。cue是人工实验指令，只进入脑：left及cuts在ANN.6—3.6秒+1，right−1，empty/cuezero0。这里没有视觉图、气味或目标坐标输入，也不生成刺激假图。

实际路径：当前bodyobs238取原允许111→已训练normalization／feedbackNN→实际AN身份1msqueue2msrelease→脑GRU16＋cue→DN2身份队列→已训练decoder→冻结combinedmotor256→42有限PD。旧低层walking本身持续步态，不将其启动／停止归给新脑或颈部。AN08B018与任意111身体感觉之间的映射属于工程学习，不是其native振动／本体感觉调谐的生理证明。

cut于ANN1.6秒，即全程7400ms的5ms决策之前执行：set_cut丢弃该方向在途包、heldport变显式无量纲静默0、停止该方向未来发送，其余方向继续；脑16／motor256／时钟不reset。DN0送入learneddecoder实际非零输出保留，绝不直接turn0或action0。不是生理切断等效验证。

## 全程时序与交接

总1＋4.8＋9.6=15.4秒，原生12.5us，共1,232,000步，1ms身体15400记录。5ms神经3080记录包含settle200、teacher960、ANN1920，状态布局128＋64＋64＋16=272；AN1、DN2缓存和queue分别记录，不凑神经元数。

从t0开始每5ms用实际body111编码AN，队列每1ms推进。settle实际动作0／motorhidden初始；teacher阶段原StaticTeacher真实驱动，motor在command0下连续priming960次，脑hidden保持0、DNzero入队。warmupcommand0为显式预热，不冒称由decoder产生；同时保存shadow_decoder_turn。

交接t5800ms不reset任何物理、motor／brain状态或queue／heldport。首ANN实际heldAN来自5795ms采集／5797ms发布的真实预热反馈，和离线初始AN0的历史不同，这是新闭环物理需验证的范围，不为匹配离线强行清队列。新brain第一ANN先用heldAN计算DN，当前动作使用已发布heldDN→decoder，符合单5ms决策延迟。

发布release_step与实际consume_step独立保存；1ms发布2ms后可在5ms决策消费。summary的恢复窗ANN4.12—4.62是纯描述性固定比较窗，**不是视觉20ms消失发布3.62秒**；本实验cue在3.6秒撤除，受脑状态与源队列影响的实际回落由数据决定，不造统一起点。

## 完整性审核和失败恢复

每条件保存完整1ms/qpos/qvel/COM/loads/nonfoot/actions／geometry、3080X238／272状态链／AN-DNencoded和held／实际capturerelease／cutflags／cue／shadowturn／appliedcommand／motorhiddenmeansactions，以及SOURCE_QUEUE完整传输和消费事件。无相机、装饰渲染、额外physicsforward或根姿态写入。

新进程audit从保存的实际X238重编码、实际源queue、脑和decoder到motormean/action及所有state逐位重放，重算原5物理门、周期、native步数／非足／额外力，并核实际handoversnapshots/sourceSHA。预热物理与原neutral基线比较只作实际核验，不为了贴合历史强改状态。

原5/slip/all6/NF/extra与转向效果分列；不预设六回合功能ALLPASS。matchedcuezero与empty、normalL与cut1.6前缀以及cut后效果真实记录，不事后加阈值。JOBS每子PID/exitcode；出错先保存partialartifacts，完成的物理不因汇总或审核错误盲重跑。whole-neck、nativeDN／AN功能、脑控制启动停驻、电极及意识结论仍未建立。
