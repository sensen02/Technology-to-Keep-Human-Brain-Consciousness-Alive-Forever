# 融合实时观察聚合修复：ONLYCODE已完成

实际标准库check：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T064313Z/fusion_aggregation_repair/codecheck.py`
已由子运行exit0，41项全部通过；没有NumPy／Torch导入、数组payload读取、模型加载、forward、训练、Core／物理／render或网络。旧已完成的融合、感官和身体源均未改。

## 先绑定实际完成的新嗅觉权重

新嗅觉250权重未知时拒绝运行dataset或预检。父需要先实际完成并独立审核新目录：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T064313Z/odor_memory_repair/`

公共规范为common.FORMAT＝ODOR_NEURAL_OFFSET_REPAIR_V1，common.load_checkpoint(path,digest,final=True)返回odor／payload，严格final250；绑定器检查实际7stage退出码、TRAIN RESULT updates250／sha、VERIFY_000250及EVALUATION RESULT／VERIFY、FROZEN源／数据身份。不会从猜测SHA或空占位准入。

```bash
python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T064313Z/fusion_aggregation_repair/prepare_odor_binding.py --bind-completed-odor --odor-sha256 ACTUAL_ODOR_FINAL250_SHA
```

该命令只标准库核metadata／bytes，写PARENT_ODOR_BINDING.json，不加载模型／数组。

## 父唯一CPU1执行融合校准

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T064313Z/fusion_aggregation_repair/run.py --execute-parent-fusion-repair > /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T064313Z/fusion_aggregation_repair/PIPELINE.log 2>&1
```

父用托管后台job管理，串行7stage：preflight／train400／verify0／verify200／verify400／evaluate／audit。每stage单独PID、日志与exit；失败停止，不盲目重跑、加步或缩批。运行目录输出exclusive，保留所有旧失败。

## 固定来源和训练范围

24训练序列＝最新045302实际14身体＋已通过033622实际视觉normal6＋4条显式双方神经输入零辅助。前20条视觉均由实际capturedRGB重放冻结99698，要求与storedheld16／h32／rawturn／latent逐位相同；新odor250使用相同实际actorreceptors重放，输出预期不同于旧66f，绝不写旧嗅觉输出精确保持。旧视觉6条没有嗅源，所以输入嗅觉网络真实零receptor，自然NN偏置保留，不将odorraw gate0。

四个ZERO辅助只保最新Fleft／Fright／Oleft／Onoodor的实际111身体观察，显式将两个脑侧感觉神经输入都置零，target0；不生成伪RGB、感受器记录或新身体counterfactual。

V教师＝实际视觉NNraw，O教师＝新嗅觉NNraw；只有F教师在离线数据生成时按强度加权desired并截幅±.78，F_DNcut仍用正常联合signal作为teacher、条件cutflag不进入脑或loss。所有y[k]=desired[k−1]／首0，data精确检查1920×24×1，不重复旧3839标签错误。训练推理模块没有融合公式／clamp／gain／sign／runtimeclear/reset。

strictdb25融合全statecopy，脑输入3／GRU16、DN2／AN1、原111掩码／normalizer不变。视觉99698、新odor250、反馈／重建／normalizer冻结，仅10脑／out／decoder张量训练。effbatch24由3个micro8完整1920累计一次Adam，freshAdam3e-4／固定400／save0,200,400最终400，不追加预算或缩批；wall1800秒／RSS1800MiB。

预检必须实际核parentdb25初始计算parity、完整24Python vs3micro8损失／10梯度／Adam、冻结tensor、传感加载器跨namespace alias恢复、真实1ms来源队列9600步／2ms发布／5ms消费。保存重放在新进程验证0／200／400生产／真实队列／RNG／10Adam计数／源身份，最终24＋2旧视觉控制指标独立审核；全部是已知归档观察，不是新身体通过。

## 公共API供身体子使用

格式：SOURCE_NECK_MULTISENSORY_AGGREGATION_REPAIR_V1。

common.load_checkpoint(path,digest,final=True) → (bridge,payload)，strictfinal400。

common.load_sensory() → (vision,newodor,visionpayload,odorpayload)，读取实际PARENT_ODOR_BINDING。

NeuralBridge.brain_step(deliveredAN,modalities,h)中modalities为(B,2)固定[visualraw,odorraw]。model.identity的newodorSHA由initialize设置instanceattribute，不动态importcommon，避免身体加载器切换alias后碰错误模块。脑＋两感官＋motor递归合336，cache／filter／pending／ports另记。

最终权重必须父实际完成400与独立审核后取真实SHA再准入新14身体；当前没有新身体代码或执行。旧14中的Fleft周期FAIL、嗅觉cut持续偏置FAIL、原滑移全部FAIL及旧D确认失败保留。修复训练不改变任何身体阈值、物理参数、嗅觉alpha或颈部结构身份，也不证明原生嗅觉功能、食物到达或意识。
