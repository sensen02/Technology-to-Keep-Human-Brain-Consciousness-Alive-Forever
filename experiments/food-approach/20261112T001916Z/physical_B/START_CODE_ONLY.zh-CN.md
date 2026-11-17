# 候选3成对batch行为权重：新B6身体确认工具

状态CODE_ONLY_READY，尚无准入。仅运行check_codeonly.py标准库检查，exit0/20checks PASS；未importTorch/NumPy/MuJoCo/模型、未加载数组/权重、未forward/Core/仿真。INPUT_BINDINGS.json、FROZEN.json、PREFLIGHT.json及trials尚不存在。候选3最终SHA与离线全部通过结果未知，不预先绑定。

全部本次文件仅位于 `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T001916Z/physical_B/`，未修改候选2未运行草稿、候选3核心、旧冻结AB或主tracker。

## 最小复用与两个协议修正

- prepare/preflight/orchestrate仅克隆候选2未运行草稿的受限函数codeobject并显式新ROOT；runtime/runner/audit完整newnamespace。没有复制整套物理算法或泛化schema框架。
- 准入期与每个新runtime加载前，`bindings.inherited_v2_proof()`核验并展开原AB的FROZEN和FROZEN_SUPPLEMENT，核原freeze与修订的捕获SHA，再明确验证同motor789f的VERIFY_neutral完整性、工程标志、condition/run、RESULT和各baseline数组artifactSHA。补充冻结、基线证明及全部引用都进入新FROZEN；不是只信原FROZEN.json。
- summary取冻结 `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T224234Z/summarize_B_v2.py` 的`build_source()`生成确切v2函数，再绑定newROOT与完整globals。已标准库抽取纯build_source验证生成源码可编译，保持effective receptor onset起算2秒和unique_physical_trials术语；不会回退source onset的v1。

## 准入和模型次序

新common实际文本确认FORMAT=`ODOR_PAIRED_BATCH_RECOVERY_V1`，`load_checked(path,digest,require_final=True)`只部署final500；shape仍2受体/GRU32。prepare保持既有all12配置全部flags通过＋独立VALIDATION/VERIFY结果及权重hash绑定＋TRAIN/VERIFY_000500保存审核，严格拒绝任何失败，不猜SHA。

原A3仅以A_GATE/result/audit和motor789f exact proof重用，不重跑。新B6所有回合使用新行为权重，包括新的B_noodor；刺激包络、heldC、causalpending/filter、288连续状态、head几何、原身体与指标门均不改。没有runtime bias subtraction、deadzone、reset或新动作门。旧motor资格FALSE和用户接受基线的experimental属性继续原样记账。

新sensor strictloader先加载，再motor严格加载，最后physical pin/Core；原guard不改。函数globals是完整显式重绑，不只给runtime属性赋值。真实loader/namespace预检只能父在资源释放后运行，代码检查不冒称模型实测通过。

## 父精确执行顺序

仅在候选3全部离线目标及独立审核已实际通过、且父知道完整实际final500SHA后，替换下面`ACTUAL_FINAL500_SHA`。这一步标准库核证后只生成新准入/冻结及字节相同的原A_GATE/CLAMP_LEVELS副本，不产生Core或物理：

```bash
python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T001916Z/physical_B/prepare.py --bind-completed-offline-pass --behavior-sha256 ACTUAL_FINAL500_SHA
```

若任何offline gate为FALSE或文件缺失立即拒绝。禁止改旧权重/VERIFY使其通过。准备成功后，父统一独占heavy做strict final500模型与newnamespace预检：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T001916Z/physical_B/preflight.py --parent-resource-released
```

allpass后父用managed后台顺序启动（一个CPU1重进程，父收集job；本子不启动）：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T001916Z/physical_B/orchestrate.py --parent-resource-released
```

B顺序noodor/equal/leftpulse/rightpulse/LR/RL；每个完整1s settle+4.8s teacher+9.6s ANN实际，随后独立新进程审核；partial output拒绝盲重跑，完整结果复用前核新behaviorSHA/RESULTaudit绑定。全部6项后输出本scope的B_FUNCTIONAL_SUMMARY.json，schema与ABv2一致，可供后续新visual轻适配。

结果目录：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T001916Z/physical_B/trials/<B_ID>/`

新汇总：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T001916Z/physical_B/B_FUNCTIONAL_SUMMARY.json`

检查证据：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T001916Z/physical_B/CODEONLY_CHECK.json`

当前没有为未知权重创建视频框架或复制旧visual；新实际trial与审核完成后，父可新轻入口按INPUT_BINDINGS行为SHA与ROOT适配，旧ABvisual冻结源码保持。本工具生成本身不代表候选3离线通过、刺激响应修好或身体确认完成。
