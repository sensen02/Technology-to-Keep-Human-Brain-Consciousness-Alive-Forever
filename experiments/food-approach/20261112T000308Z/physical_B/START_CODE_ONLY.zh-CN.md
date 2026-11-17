# 候选2新行为权重的B6身体确认：CODE_ONLY交付

仅准备完成，未运行任何新权重加载/forward/数组/Core/物理。实际只执行标准库check_codeonly.py，exit0/22项全部通过，记录同目录CODEONLY_CHECK.json。当前INPUT_BINDINGS.json、FROZEN.json、PREFLIGHT.json和trials均未生成，不存在提前准入。

## 严格准入

先等待候选2固定500训练、全部12个离线配置功能目标及独立验证全部通过。prepare.py显式要求父提供实际SHA，不猜摘要；检查新checkpoint路径、VALIDATION/RESULT.json all_targets_pass、每行flags、VALIDATION/VERIFY.json all_targets/integrity/resultSHA、TRAIN/VERIFY_000500.json actualsaved审核，以及motor789f身份。任何False或缺失立即拒绝，不跑身体。

A3只引用原AB224234Z的A_GATE与三项RESULT/VERIFY并核same motor SHA和各原哈希，完全不重跑A。CLAMP_LEVELS和A_GATE复制为本scope独立只读字节相同输入。B_noodor必须用新权重新跑，不能借用旧sensor的noodor。其它5B也各新完整实际一次。

## 最小适配

runtime.py导入旧冻结ABruntime后，将每个继承函数按新globals字典克隆，显式替换新ROOT/SENSOR/SHA/load_policies/source_paths/verify_frozen；runner/audit/summary入口同样克隆冻结原codeobject并绑定完整新globals。不是只改runtime变量却遗漏from-import-star复制值。旧物理代码逻辑保持，使用新PREFLIGHT/FROZEN与新目录。schema仍STIMULUS_AB_288_V1数据布局，source/INPUT_BINDINGS明确候选2ODOR_SEGMENT_RECOVERY_V1权重与新身体scope；旧motor资格旗和A/B原门保持，不改旧strict qualification。

新load_policies首先严格candidate2 common.load_checked final500，确认newformat/step/same32memory，再strict motor加载，然后runner才pin physical/Core。sensor旧guard不修改。preflight.py的模型加载/全局绑定核验也仅由父在资源释放后执行，不进入Core/积分/训练。runtime/source映射和所有旧、新source/data、候选离线证据、原A引用全部在运行前绑定FROZEN，执行前后验证。

新B6继承原288神经链、5mscausalpending、刺激on/off/反转时序、0.2s包络、完整1+4.8+9.6、71opaque、原全物理模型/执行器、nativeNF摘要与独立actualaudit。SUMMARY精确保持原rows trial/functional_pass/flags/metrics及24条artifact sourcehash schema，新Bsummary写本physical_B目录。没有受体切断或空间导航声明。

## 父执行命令（当前禁止本子执行）

第一步只在最终离线PASS且已知真实SHA之后运行；`ACTUAL_FINAL500_SHA`由父替换成完整实际64位摘要，不能填旧4be0/bfd6或任意值：

```bash
python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T000308Z/physical_B/prepare.py --bind-completed-offline-pass --behavior-sha256 ACTUAL_FINAL500_SHA
```

成功仅生成INPUT_BINDINGS/FROZEN和原A/CLAMP字节副本，没有forward/Core/物理。下一步父独占heavy，用既有环境做strictload和新namespace模型ABI核验：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T000308Z/physical_B/preflight.py --parent-resource-released
```

PREFLIGHT allpass才运行串行新B6，父必须managed后台并收集job：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T000308Z/physical_B/orchestrate.py --parent-resource-released
```

orchestrate逐trial运行runner→新进程audit，已完成结果只有完整身份核验通过才复用，遇到部分trial拒绝盲重跑；六项后运行新summary。新result目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T000308Z/physical_B/trials/<B_ID>/`，summary `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T000308Z/physical_B/B_FUNCTIONAL_SUMMARY.json`。

现有旧visual硬绑bfd6行为SHA与旧ROOT，不能直接指向新身体。新B6实际通过独立审核后，需要新visual小入口绑定本scope的INPUT_BINDINGS行为SHA与ROOT，renderer物理/288/因果/摘要逻辑不改，旧ABvisual源码不动。此处尚未实现新visual，不冒称已输出图或视频。

## 待父/核心实测的边界

1. 新prepare对VALIDATION schema沿用候选2实际adapt_source.py所继承的234150格式，已文本确认；尚无真实PASS结果，不假定会通过。
2. CODE_ONLY仅验证AST、模块顶层无重imports、显式函数globals克隆、final500/12目标/独立门、B-only及未知SHA未生成准入。真实strictloader与old-codeobject绑定必须父实际preflight证明，失败先保留，不在已运行freeze内热改源。
3. 原A_GATE B_executed:false仅旧历史字段，重用说明明确A_motor_gate_reused而非本scope又跑A。不得修改旧A_GATE或把这个字段改成true。
4. 不改变任何候选2核心源码，physical_B是子目录，不纳入当前训练source_paths显式文件列表；无主tracker写入，无heavy任务或临时失败碎片。
