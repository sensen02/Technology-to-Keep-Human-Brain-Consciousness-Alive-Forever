# 有条件 postrun final900 batch 审核（ONLYCODE）

本工具只接受未来真实状态 `ORIGINAL_PIPELINE_FAILED124_SAVED900_POSTRUN_QUALIFIED`，不替代或改写原三阶段成功路径。若原 wall_replan_9000 三阶段全部成功，这套兼容工具拒绝使用；若原仍RUNNING、只到750、缺900、postrun尚未完成、错误FORMAT、缺completion或有FAILURE，都不能准入。当前实际postrun是否合格与900 SHA均未知，本次不执行科学计算或任何postrun准备。

科学模型和所有 common/model/data/scan/learning/utils 始终是原目录：

`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T200608Z/state_aligned_action_calibration/wall_replan_9000/`

资格证明来自：

`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T200608Z/postrun_saved900_qualification_ONLYCODE/`

后者没有model.py，绝不能把它当模型producer。原strict common.load_checkpoint(actualOriginal900, actualSHA, final=True)不绕过。公开alias/path仍沿冻结审核真实上下文finally恢复，不clonecontextlib wrapper或任何科学函数。

资格严格读取 actual postrun metadata.qualify：原terminal FAILED124、无RUNNING、原600/750各81fresh通过、恰601..900的300日志、实际原子900字节SHA。随后要求postrun三阶段FRESH900_SAVED_AUDIT/SAVED900_EVALUATION/INDEPENDENT_SAVED_METRICS全部COMPLETED/exit0，fresh900全81、只读严格payload observer证明FORMAT900/8Adam计数900/curve300精确/anchorRNG原strictloader，EVAL与独立NumPy指标、全部stage receipts与3600s/RSS1800资源completion、无FAILURE。所有原600/750/900 PT、original JOBS FAILED124、原日志及新postrun证明进入before/after SHA。原始失败保留，不声称original_pipeline_all3exit0。

两个来源域必须区分。postrun TRAIN/RESULT source_hashes包含新qualificationtools，比权重config更宽；它不是actual_model_source_hashes。actual maps由600/750/fresh900保存proof一致性约束，绑定fresh900原strictpayload source_hashes/data_hashes，并与postrun EVAL对应；返回给冻结audit的tr是明确标识的内存derived view：source/data换成实际payload maps，同时另存qualification_source_hashes/qualification_data_hashes、derived_view_fromproof路径/SHA及两域描述。整个postrun TRAIN RESULT原文件及其wider maps都纳proofbeforeafter，不改文件/PT，不宣新wrapper在旧权重config里。VERIFY和SOURCE_LEDGER显式记录derived view及两source域，qualification_basis明确为原FAILED124后合格，不假称原三stage成功。

科学调用复用已冻结 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T200608Z/final_state_aligned_batch_audit/audit.py`，SHA ceb7f5838efe6adb0439db9ce4ca051f4d7af3d7415752595088d67270779b2e；contracts基线SHA 2b3695aef31f1192434b02b2f7308ca077bb6ea300e50f2712c7eaf6de78e0bf。新audit精确AST仅改四个资格/资源文字常量、添加两项资格source-domain元数据记录；contracts仅替换资格函数、固定原producer与新输出ROOT、WALL1800和资源文字。inverseAST标准库检查要求其余科学节点完全相同。

所以72组仍原样：conditional36组六输出及actualTF36组真实四输出，全部12×1920、B1/B12/三micro4、独立fullproduction参照、误差≤1e-12、old256exact、finite，前后weights/RNG/source/proof/checkpoint身份。原TF/conditional标签语义、NNloops、矩阵/tanh顺序全部不动。0optimizer/backward/训练/Core/物理；审核真运行时有神经调用，不能标0。本轮ONLYCODE则Torch/NumPy/数组/PTpayload/模型调用全部0。

本新scope首次READY前预登记 **1800秒／RSS1800MiB／单CPU**，intra/inter-op与OMP/BLAS/MKL均1，在线程创建前pinCPU。依据实际新600完整publish约2200秒含多次strictv4、新900单load需多次严格祖先加载及额外72比较/哈希，是新工具跑前预算而非已测postrun审核时间。不修改旧600／900审核的600／300预算、原pipeline9000或当前其他任务；1800超时如实失败，不扩预算、不缩样本、不绕strictloader。

仅在真实postrun资格成功并由父取得实际900 SHA后，父独占managed运行并collect：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T200608Z/postrun_final_state_aligned_batch_audit/audit.py --checkpoint /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T200608Z/state_aligned_action_calibration/wall_replan_9000/TRAIN/checkpoint_000900.pt --sha256 "${ACTUAL_POSTRUN_ORIGINAL900_SHA256:?父须先取得实际原900SHA}" --execute-parent-final-state-aligned-batch-audit
```

新目录独占输出RUN_<actualfullsha>/，存在拒覆盖；VERIFY/ROWS/SOURCE_LEDGER/REPORT/RESOURCE_COMPLETION全部沿原失败失效机制。父最终接纳需实际进程exit0、VERIFY.statusPASS/qualified_final900、completion.pass、四artifact SHA重读一致、无FINALIZATION_FAILURE。contracts.accept_run仍用此新1800预算执行实际readback。任何失败不修改原资格或旧FROZEN。

纯STD检查：

```bash
/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T200608Z/postrun_final_state_aligned_batch_audit/staticcheck.py
```

检查冻结baseline SHA、精确inverse科学AST、globals/contextmanager、原producer身份、实际model/widerqualification source域分离与beforeafter三PT，以及12个未来metadata负例（RUNNING/原all0/缺900/格式错/postrun未完/无completion/fresh81失败/Adam或curve错误/假原成功/源域误混/completionfalse）、1800边界。仅内存mock与stdlib源码，无prepare/qualify真实调用、Torch/NumPy或数组/PT/模型加载。ONLYCODE_READY不等于postrun已成功、数值审核PASS、body／视频／真实颈部通过。
