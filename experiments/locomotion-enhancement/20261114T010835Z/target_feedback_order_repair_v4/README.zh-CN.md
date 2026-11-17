# v4唯一资源修订：参考预检3072MiB，训练／保存审核仍1800MiB

v3实际预检在最后resource_gate因fullB12逐B1 autograd参考图峰RSS超过1800失败，未写PREFLIGHT，不能据此称数值检查已通过。原v3日志／源码／输入保留不改，本scope是跑前阶段资源重新登记，不追改旧结果。

唯一变化：fullB12reference_preflight_only允许3072MiB、5400秒；实际续训与保存审核依旧learning.resource_gate原1800MiB／5400秒。模型／FP顺序／loss／mask／float32／精度门／原300Adam-anchor-RNG／剩300更新／300450600保存全部不变，没有activation checkpointing或隐式放宽。

preflight.py读取冻结v3完整AST，仅从learning import移除resource_gate一项，使原main该全局调用明确绑定本文件preflight_resource_gate3072；数值语句不改。PREFLIGHT额外明确resource_scope='fullB12reference_preflight_only'、preflight_RSS_limit_MiB3072、training_RSS_limit_MiB1800和实际peak。train/eval/verify没有导入该helper，learning文件与v3逐字一致，3072预算不能泄漏进训练。

父metadata准备绑定v3全部源及failedRSS日志、INPUT_BINDINGS与无PREFLIGHT事实，然后一次实际预检。若再次超过3072或数值失败，保留证据、先报告，不继续无限提高预算。

```bash
/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_order_repair_v4/prepare.py --prepare-resume300-order-repair
```

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_order_repair_v4/preflight.py --execute-parent-order-preflight
```

仅actual完整all_passed／READY_RESUME300_TO600后恢复301..600：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_order_repair_v4/run.py --execute-parent-order-resume
```

本子ONLYCODE未导数值库、读取数组、创建模型/Core、训练或物理。完整预检门须父实际执行；预算修订本身不代表数值或行为通过。最后600仍需同四项无教师身体、真实DN2decoder/fullintegration后续验收。
