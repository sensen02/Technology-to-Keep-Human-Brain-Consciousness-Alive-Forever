# 独立 final900 状态对齐动作模型 batch 审核（ONLYCODE）

本目录只审核真实 producer `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T200608Z/state_aligned_action_calibration/wall_replan_9000/` 的最终 checkpoint_000900.pt。实际新900 SHA 尚未知，必须由父在真实三阶段完成后显式提供；代码不推断 SHA，不把祖先600或750当最终900，也不改 producer、当前长平地任务或旧审核/诊断源。

模型格式按 producer 实际源码为 `INDEPENDENT_MOTOR_STATE_ALIGNED_ACTION_V1`，严格调用该 producer common.load_checkpoint(..., final=True)。三保存点是新600、750、900；三主阶段为 RESUME601_TO900_AND_FRESH600_750_900、FINAL900_STATE_ALIGNED_EVALUATION、INDEPENDENT_SAVED_METRICS。全部必须真实 COMPLETED/exit0，TRAIN总900=原600+新300，三份fresh VERIFY、对应checkpoint与conditional arrays、最终实际/conditional评估及独立EVALUATION_VERIFY均按实际schema/SHA核验。所有600/750/900 checkpoint文件都进入前后proof ledger。

资源准备资格接受 producer 的真实流程：原132数值预检96项正确性全部TRUE，但原all_passed=false/timingblocked7200保留；新scope通过科学源码逐字/资源AST等价和训练前wall9000重登记接受该已有正确性。新PREFLIGHT中的 new_numeric_preflight_runs=0、new_NN_forwards=0 是合法、明确的来源说明，不要求重新数值预检。原false证据、metadata路径/SHA、复用96内容、science_equivalence及RESOURCE_REPLAN全部核对。不要运行 producer 复制归档的preflight.py。

数值部分只在显式父入口、完整最终资格通过后进行，全部torch.no_grad，不创建优化器、不backward、不训练、不构建Core或物理后端：

- conditional：12条×1920tick，以独立 public fullproduction_feedback 为参照；B1、全B12、三micro4各case六输出 mean/residual/h96/pretarget/posttarget/action 每项误差≤1e-12、全部finite、old256与来源cache逐位exact，共36组六输出比较。
- actual TF：producer真实接口 tf_forward/make_actual_scan 输出四项 mean/residual/h96/action，以独立 fullproduction_feedback(forced=True) 的真实对应项作实际source参照；相同B1/B12/micro4覆盖，共36组四输出比较。forced实际pretarget由data.prepare按原来源编码合同构造，不发明TF额外target状态或监督。
- 两条分支都保留 producer 原矩阵／tanh／target积分运算，不在审核里修正被测实现。actualTF动作监督与conditional仅target/prior/anchor的语义不变，conditional action距离仅COUNTERFACTUAL_DISTANCE_NOT_SUPERVISION。

显式保存并finally恢复八个public alias和sys.path：adapter/sources/common/model/learning/scan/data/utils。producer本身没有adapter.py，因此仅从该producer导入其真实七个module，不强行制造或导入不存在的adapter；其祖先adapter由严格producer loader自己的明确上下文处理。所有contextmanager保留标准库原globals，不克隆装饰器。

本次新900审核在FIRSTREADY前由父预先登记 **600秒、RSS1800MiB、单CPU**。依据是旧600审核实际200–232秒且新900增加严格600祖先层和actualTF整批审核，并非已测新900运行时间。严格模型加载、全部样本/前后哈希和输出都计入本预算；600秒到即失败，不自动延期、缩样本或绕loader/source。旧600模型审核和FD的300秒记录完全不修改。

脚本在创建watchdog或导入Torch前先pin单CPU，使后续线程继承；之后枚举所有live tid要求相同mask，intra/inter-op及OMP/BLAS/MKL均1。600秒SIGALRM与1800MiB RSS watchdog保持到所有主要输出及提交回读门完成。前后核checkpoint/source/proof/weights digest/RNG，oldweights冻结，所有参数.grad保持None；实际神经执行不会标0，记录prepare/scan/fullproduction调用。

父唯一实际命令（父须先把变量设为真实最终900 SHA，且所有生产阶段结束后独占managed运行并collect；本子不执行）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T200608Z/final_state_aligned_batch_audit/audit.py --checkpoint /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T200608Z/state_aligned_action_calibration/wall_replan_9000/TRAIN/checkpoint_000900.pt --sha256 "${ACTUAL_FINAL900_SHA256:?父必须先赋实际最终900SHA256}" --execute-parent-final-state-aligned-batch-audit
```

输出独占新目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T200608Z/final_state_aligned_batch_audit/RUN_<实际完整SHA>/`。目录已存在即拒绝，不覆盖旧run。保存VERIFY.json、ROWS.json、SOURCE_LEDGER.json、中文REPORT.zh-CN.md及RESOURCE_COMPLETION.json。

最终只可接纳：脚本exit0 + VERIFY.status=PASS且qualified_final900=true + RESOURCE_COMPLETION.pass + 所有completion artifact SHA实际相符 + 无FINALIZATION_FAILURE.json。contracts.accept_run会实际回读并核这组条件，不仅检查布尔标记。收尾失败先保留failure marker，再仅在本次独占新RUN内原子更正VERIFY为FAILED/qualifiedFalse和completion.passFalse，避免残留落盘通过标记；临时JSON立即清理，不碰旧RUN或source。预算信号后的最小失败记账不授权继续任何模型计算。数值batch通过不等于body、真实颈部或全任务通过。

纯标准库自检命令：

```bash
/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T200608Z/final_state_aligned_batch_audit/staticcheck.py
```

仅AST/globals、原contextlib wrapper、真实final900/600750900 schema、内存metadata正例/拒绝例（包括合法96复用、新numeric runs0、旧false证据、缺TF门/错9000/阶段未完）、正常/异常alias恢复，以及临时自有JSON提交/原子失败更正/产物篡改与600秒预算拒绝fixture。临时文件当场清理，不接触未来producer payload、数组或当前JOBS，不导入Torch/NumPy。ONLYCODE静态通过与未来实际数值判定严格分开。
