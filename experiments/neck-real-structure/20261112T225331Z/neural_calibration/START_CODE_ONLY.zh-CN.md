# N2DNp09＋M1AN人工脑离线工程校准：父执行说明

当前仅CODE_ONLY实现完成。实际标准库codecheck.py exit0，30项入口／源文件／字段／无重导入检查全通过；子代理未导入NumPy／Torch、未读数组、未构造模型或进行forward／训练。所有执行源自该检查后停止编辑。

## 唯一父执行命令

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T225331Z/neural_calibration/run.py --execute-parent-calibration > /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T225331Z/neural_calibration/PIPELINE.log 2>&1
```

托管为唯一CPU1重任务。JOBS.json为exclusive初始，所有stage日志open x，失败先查实际日志，不盲重启／覆盖。Stages：PREFLIGHT→TRAIN→VERIFY0→VERIFY100→VERIFY200→EVALUATE→AUDIT。主入口不带显式flag只显示CODE_ONLY，不执行重模块。

## 实际来源与数据契约

- 固定源绑定 `/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T225331Z/dnp09_interface/SOURCE_BINDING.json`，SHA9e88a04dd91aa6f9c1c1c68e0562325205e0f36f5cdd9ae41d1dd46c8171dc23；严格2DNp09原序433155799／566493282、1AN08B018原序351919408。真实geometry身份与parent端口数值检查／时序日志纳sourceledger，不猜神经维数。
- 训练图像身体轨迹不是本轮新仿真：使用旧opponent7实际6回合的neural_decisions.X238，suffix960:2880原1920条自主决策，允许局部身体索引0:42／84:126／168:195共111；排关节误差、执行目标、上一动作和目标速度。没有这些原始字段就拒绝，不从动作或世界几何反推观察。
- 训练4回合empty／center／leftpulse／rightpulse；开发2回合LTR／RTL。已使用过的归档数据，不能称新物理留出。
- Cue由实际已发布CAPTURES与used_stimulus_code校对，0off／+1left／−1right／0center，明确是实验任务指令，仅脑侧输入，不是原生视觉神经或电极记录。teacher raw_turn仅作损失标签，不是模型输入。

## 架构与时序

人工脑离散GRU16，输入只有延迟AN1＋cue1，输出DN2tanh；身体解码器输入只有延迟DN2，经16tanh到turn1tanh；反馈编码器只用局部允许111，经32tanh到AN1；辅助AN1→32tanh→111重建头仅训练用。脑不读raw111／teacher，身体解码不读cue／脑hidden，无42关节输出。没有加载motor256或身体物理。

源队列工程dt1ms／delay2ms，两方向各有FIFO；brain/body5ms消费，因此每方向一决策延迟。真实production用SourceTransport推进9600个1ms tick，记录queue_release_step与neural_consume_step分开(例capture0/release2/consume5)，不改原journal，初始deliveredAN／DN为0。scan中旧AN／旧DN一决策hold必须与production逐值一致。脑记忆整个1920历史不detach、不按刺激reset。

损失为turnMSE＋.1 normalized111重建MSE＋.001DNactivity²。均值／标准差仅训练4回合计算、std下限1e-6；新数组逐sourcehash绑定。固定seed20261112／freshAdam1e-3／全部参数／full1920batch4／200更新、clip1，保存0／100／200，仅最终200开发验证；不追加步数／挑100，wall1800s／RSS1800MiB。

## 实际预检和独立重放

- full1920batch4独立Python和JIT scan的5类输出／损失／所有参数梯度／Adam状态比较，容差1e-10，检查mask、模型端口、finitegradient和内存；仅局部一次Adam预检，正式train重新seed初始化。
- batch1实际源队列9600tick与同模型rollout对比，3838次神经消费事件时序明确。
- 新进程0／100／200保存重放核全部输出、实际队列、全部Adam步数／RNG／参数实际变化／sourcehash。只final200报告训练4及开发2的RMSE／active符号／中性RMS／重建误差，并用独立scan重算结果。

不提前设验收MSE阈值；数值审核通过只证明工程离线链和记录一致，不证明新身体能行走、脑确实利用AN、DNp09原生转向规律或AN关节编码。现AN08B018直接感觉伙伴主要club，不能将学习编码器称原生关节感觉。所有结果标full_neck_training_allowed=false／new_walking_verified=false。
