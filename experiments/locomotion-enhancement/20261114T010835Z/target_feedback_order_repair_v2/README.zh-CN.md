# 冻结分量浮点次序修复：从300继续，总计600

本scope只修训练缓存／扫描的浮点次序，不改变412部署网络、352状态、8参数、dtypes、mask、损失、5400秒／1800MiB预算或1e−12门。原failed300VERIFY／JOBS／日志、original0PASS和actualconditional_parity证据原样保留，不重跑前300更新。

缓存按每tick每序列B1保存base neuralhead42、negative*bound*RIGHTcorrection42、positive*bound*LEFTcorrection42；严格保留源乘法左结合。scan旧均值用((basehead−gain*predoffset)+weightedR)+weightedL，避免先合并／加回prior再相减造成double尾差跨float32量化中点。newencoder／GRU／head也每序列B1执行再catbatch，梯度仍按完整B12／micro4×3汇总，部署不改。

恢复父checkpoint300 SHA b562fcc3e39c2aa56381a87b17f1e80124f203fa1a73e3cdf8888778de119f14；新model仅附RESUME300_ORDER_REPAIR_TO600身份，权重深复制精确。Adam全部8状态／计数300、原config.initial_anchor、原RNG都继续，setup后显式restore RNG；不重seed、不fresh训练状态、不重新锚定300权重。新更新严格301..600共300次。保存新300（初始数值资格）、450、600；旧0PASS和旧300FAIL另列，不抹历史。

strictloader读取旧recovery wrapper时，整个加载上下文显式绑定旧adapter和旧path并finally恢复model/common/sources/adapter/path，防止套错新adapter。新save与load都使用同一newlineage model身份。预检Adam full／micro／restore各用独立deepcopy原300快照，最后核parent payload未污染，不能因Tensor存储别名假比较。

先父metadata：

```bash
/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_order_repair_v2/prepare.py --prepare-resume300-order-repair
```

再独占预检（不训练）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_order_repair_v2/preflight.py --execute-parent-order-preflight
```

预检在实际300权重上验证全12 B1／B12／micro4各case六输出对完整生产≤1e−12、old256exact，13 Torch真实target／float32编码，Python/JIT、FD既定门、完整1920 loss／8grad／Adam对照及精确恢复。原Adam探针只是临时两次更新且全部恢复，不计续训。剩300×实测更新×1.3加三次fresh完整存储审核≤5400才READY_RESUME300_TO600。

通过后唯一续训：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_order_repair_v2/run.py --execute-parent-order-resume
```

三stage：300初始fresh保存审核／301..600含450与600fresh审核、final评估、独立NumPy指标审核；evaluate原proofloop已仅改300/450/600，数值公式不改。所有新结果在本scope，strictfinal600两返回API保留，旧权重／失败不覆盖。完成后仍须无教师body4与脑—realDN2decoder/fullintegration，不以条件性loss或数值修复叫运动通过。

本子只源码／JSON／字节hash／标准库检查，0Torch／NumPy／数组／模型／Core／实际训练／物理。actual全12数值、JIT、资源与续训资格待父执行。
