# 一个有限fastpath联合修复候选：ONLYCODE，未训练／未验证行为

本候选位于新时间戳20261113T195245Z，所有20261113T161918Z旧科学源、输入清单、权重、失败及已经执行的诊断不改。实际bound6探针 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/independent_motor_failure_diagnostic/REPRESENTABILITY.json:6-22` 显示8条训练轨迹逐点区间外比例与MSE下界均为0，因此本候选保留bound6；这不证明共享ANN或闭环一定可学。

模型输入仍是actual physical112+oldafter256+独立task2=370。encoded=tanh(encoder370→96)，candidate=GRUCell96(encoded,h96)，h96=(1−alpha)h96+alpha*candidate，alpha仍100ms对应的.048770575499285984；新head输入concat[h96,encoded96]192维输出42，residual=6*tanh(head)，最终action=tanh(frozen789fmean+residual)。只是ANN直接读currentencoded，不引入算法gate、clock/phase/case标签或速度乘零。old789f全部冻结、合法旧投影不变，state352及未来432未部署的界限不变。

严格初始化来自actual454581a41034f92e253a2cb31215b5ddd1e09f8795ae54fb89053fd3187d1398 final600。复制全部old/encoder/GRU，head前96列parentcopy，后96列0，biascopy。仍8训练张量但head新增4032参数。迁移要求真实8训练+4已失败学生观察全1920history、相同960oldwarm/newh0下旧256与new96状态逐位相同，mean/action/residual≤1e−12，扩矩阵GEMM不冒称逐位均值相等。代码一处新模型，无多层继承验证陷阱。

新sources.py仍通过原实际8来源资格生成训练清单，并将原输入1021项身份作为冻结输入继承，不改目标。另将parent实际存储资格、阶段A完整重放与bound6理论下界证据纳入新DATASETmanifest。4失败学生输入只DEV固定观察描述，没有teacherY、不进训练，不把相同时刻旧teacher动作当纠正标签。data.py仅一层原data数值AST，显式ROOT/read/verify绑定新目录，不误读旧INPUT_BINDINGS。

联合损失：每条序列前20个tick（0..0.095s，严格不含.1s）时间权重10，其余1900权重1；总2100归一，每条序列总质量1/8且每动作除42。5个v=1来源残差均方按全1920×42×5归一，系数从.01升为.1；新8参数对迁移起点（包括skip零列）的均方anchor系数.001。首20索引仅训练损失，不进部署actor。结构、瞬态权重、保持先验三者联合改变，不称单因素因果。

freshAdam3e−4固定600、effective8／micro4×2，每micro完整1920BPTT，保存0/300/600、只final600资格。preflight先来源／迁移12条及Python/JIT／cache-production，再完整1920 B8对两批B4损失、8梯度、Adam及参数差≤1e−10；检查first20质量和excluded rawchannels直接梯度为0；旧frozen和RNG／源码前后保持。完整update计时×600×1.3再加三次fresh保存审核开销估计，总预测≤1800秒才READY，RSS1800MiB；不满足先阻断，不自动减步或延期。

训练／fresh保存验证／final评估／NumPy独立审核复用旧已实际运行入口的单层AST，只有新目录import和元数据重绑；新objective通过newlearning解析，三层alias框架不引入。存储publiccommon.load_checkpoint(path,digest,final=True)两返回，strict新FORMAT INDEPENDENT_SPEED_YAW_MOTOR_FASTPATH_REPAIR_V1、step0/300/600、8Adam计数／全部旧权重／source／RNG。保存每次live model/Adam/RNG精确roundtrip，再新进程完整8生产与缓存重放。final增加4DEV输出范数但没有teacher误差或行为PASS。

父先读源码；标准库来源准备：

```bash
/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T195245Z/fastpath_motor_calibration/sources.py --prepare-source-metadata
```

随后父独占CPU托管预检：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T195245Z/fastpath_motor_calibration/preflight.py --execute-parent-preflight
```

仅actual all_passed与timing READY_FIXED600后，运行一个有限候选：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T195245Z/fastpath_motor_calibration/run.py --execute-parent-calibration
```

流水线同原3主stage，train包含0/300/600各新进程审核。source/旧data全部冻结；只有新目录写产物，finalSHA在actual完成前未知。低fit不行为通过；新权重之后仍需同四bodycomponent门独立验证，再接brainNN→来源realDN2queue→decoder2与fullsensory回归。本候选没有物理、教师学生状态查询或neck部署；所有旧四组件FAIL、粗地形slipFAIL和运动提升目标仍保留。
