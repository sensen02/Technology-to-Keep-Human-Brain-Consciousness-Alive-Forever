# 实测驱动的跑前墙钟重登记：7200秒→9000秒，不重做数值预检

原目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T200608Z/state_aligned_action_calibration/` 的132实际预检有96项正确性全部TRUE：迁移／TF／conditional、13执行器编码、完整梯度／Adam／RNG等已实测通过。但原all_passed=false且timing_status=BLOCKED_BEFORE_TRAINING_TIMING，因为完整估计7886.235788201401秒超过原7200秒。没有新训练更新。原JSON／源码／失败日志保持原样，不追改成通过。

本独立嵌套目录仅跑前重登记wall9000秒。仍是同一41fa600权重／原Adam8／RNG／最初fcbd anchor，only601..900共300新更新，save600／750／900，模型412／352／8tensor、实际状态TF动作监督／conditional9target与6prior分离、数据mask、dtype、数值判据全部不变。训练RSS1800、参考预检RSS3072保持。

`equivalence.py`要求11个科学文件逐字一致；对learning.py／run.py／staticcheck.py只逆还原已登记资源差异后AST必须等于原代码。learning.resource_gate default9000及旧_ns公开映射都指同一9000函数，1800MiB不变；三stage只有一个9000剩余deadline，不逐stage重置。PREREG只允许resources.wall_s、preflight描述中的跑前时间阈值改为9000，以及资源重登记provenance；逆还原这两处时间元数据后须逐值等同原记录，不构成新的NN预检要求。

prepare仅标准库：校验原96项TRUE、原all_passed确实false／timing-only blocked、原完整source/data maps一致与当前SHA、全部12条件六输出0、原数值精度／FD／parent600恢复门、13integrator实际重放及RSS范围；重新由实测12.859299216070212秒更新与fresh／EVAL allowance核算7886.235788201401<9000。然后绑定原输入／失败证据及新源码，并输出新INPUT_BINDINGS／PREFLIGHT／RESOURCE_REPLAN。

新PREFLIGHT会明确reused_actual_numeric_preflight路径／SHA、oldoriginalall_passed=false、resource_replan_pretraining=true、corrected_readiness_against9000=true、qualification_basis=既有96实际正确性＋新源码等价＋新时间规划；所有数值指标标原artifact来源，new_numeric_preflight_runs=0。新all_passed仅用于新运行入口，不是重写原结果或声称新数值运行。

父资源释放后，仅运行metadata：

```bash
/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T200608Z/state_aligned_action_calibration/wall_replan_9000/prepare.py --prepare-wall-replan9000
```

成功后直接唯一续训pipeline：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T200608Z/state_aligned_action_calibration/wall_replan_9000/run.py --execute-parent-state-resume
```

不要重新运行复制的preflight.py；它仅是原数学源码等价的归档副本。新scope结果写本目录，原timingblock证据不改。静态计划检查与资源重登记不代表新模型训练或行为通过；final900后仍须无教师身体四项与真实颈部完整主线。子仅ONLYCODE／STD metadata计划干跑，0Torch／NumPy导入、0数组值／模型／NN／Core／训练／物理。
