# 反馈数据聚合修复v2：父执行说明

新候选独立目录，所有旧225331Z训练、接口、实际身体数据和失败原字节保持。当前仅CODE_ONLY；实际codecheck.py exit0／40项全部通过，未导入NumPy／Torch、读数组、构造模型或训练。执行源自此检查后停止编辑。

## 唯一父CPU1命令

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T000915Z/feedback_aggregation_v2/run.py --execute-parent-aggregation > /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T000915Z/feedback_aggregation_v2/PIPELINE.log 2>&1
```

不带flag不导入数值模块。父应managed后台运行并收集所有stage；PREFLIGHT→TRAIN→VERIFY0→VERIFY200→VERIFY400→EVALUATE→AUDIT。输出与日志exclusive，已有部分结果时拒绝盲重跑，实际错误先定位，不改预算／batch私自恢复。

## 固定变化与保持项

- 从实际finaleed1ad3cc61a1b59086ff2324175f6bf3b5da5aa197165e6c4b009c17027c95e初始化，全部模型参数、111均值／标准差、允许原索引和源9e88队列身份精确保持。人工脑离散GRU16、DN2、AN1、身体解码、反馈与训练辅助重建架构不变。
- 原4训练111输入／cue逐位保留，加入3实际P_empty／P_left_pulse／P_right_pulse的X238自主suffix1160:3080。原neural记录3080、原old视觉记录2880的suffix960，两者严格区分。不从几何、动作或理想轨迹反推反馈。
- 七条完整9.6秒训练记录，仍coldbrain0／heldAN0／heldDN0，不增加warmstart变量。完整1920历史梯度，不detach／不刺激reset。
- 所有训练和开发标签统一为实验指令cue×.8，0→0、±1→±.8。这个teacher只在离线损失使用，不进入部署控制器。原视觉／实际turn另存供描述，不再把相同符号cue配上不同retinal幅度要求。
- 每序列有cueon与neutral时各占action点权重总质量.5，否则全程均匀质量1。loss为weightedactionMSE＋.01归一111重建MSE＋.001DN活动平方＋.1decoder(零DN)平方。零DN损失是训练先验，运行时仍直接调用decoder，无clamp、bias减法或条件turn门控。
- normalizer沿用父模型，不重新fit／增floor／clip反馈；报告旧4与新3的归一输入范围。这是数据聚合、教师范式、损失和训练的联合修复，不能声称唯一原因。

## 数值和保存核验

实际完整batch7／1920预检比较Python与JIT生产输出、独立手工weightedobjective、全部参数梯度、Adam参数／moment／step、sourcequeue1ms9600tick→5msconsumer输出和时序。峰RSS必须<1800MiB，超出就报错，不自动降batch。

正式freshAdam1e-3／seed20261112／fixed400更新、clip1、保存0／200／400，仅final400验证。墙钟上限1800秒、RSS1800MiB、不加步／不择200。新进程分别核保存生产batch1／7、实际onpolicy sourcequeue、RNG(按manualseed→initialize真实父加载复刻)、全参数Adam计数以及参数更新非零和normalizer不变。

开发仍是旧LTR／RTL两条归档输入，标签改成明确指令目标；不是新物理留出。最终报告7训练＋2开发的RMSE／方向／neutralRMS／重建与实际零DN解码输出；另保留旧retinal／actualteacher差作为描述。finalmetrics新进程JIT独立重算和SHA审核后才记完整性完成，不造功能全PASS门。

## 范围限制

本流水线不运行身体、MuJoCo、冻结motor256或新的干预，不说明实际反馈自激已经修好。原empty turnRMS约.538和所有身体滑移FAIL保留，新身体6回合在最终候选后另行预登记与验证。原始DNp09生理功能、AN08B018关节调谐、完整颈部、电极或意识均未由此建立。
