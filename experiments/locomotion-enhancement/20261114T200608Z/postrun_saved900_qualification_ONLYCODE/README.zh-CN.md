# 有条件postrun保存900资格工具（ONLYCODE，非训练恢复）

目录：`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T200608Z/postrun_saved900_qualification_ONLYCODE`。
原实验：`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T200608Z/state_aligned_action_calibration/wall_replan_9000`。

只在原144真正结束、原JOBS真实最后FAILED/124截止、没有RUNNING、真实原checkpoint_000900.pt存在且父明确提供其SHA时才可执行。若原三阶段全exit0，工具拒绝不必要恢复；若只有750、900未保存、原仍运行或SHA未知，拒绝。绝不改144或延长其9000秒截止，不接缓存、不新训、不恢复优化更新，也不改既有READY body/batch/video入口。

STDmetadata严格核原阶段prefix和实际125712命令/log/elapsed deadline，原TRAINING.jsonl必须恰601..900连续300条，无重跑/额外记录；原TRAIN/IDENTITY恢复actual41fa600 Adam/RNG/originalanchor/TF监督身份、原600和750各81项fresh proof/PT SHA/conditional数组header与opaqueSHA、源数据图不漂移。原TRAIN/RESULT如果在截止前来不及导出可以缺失，若存在必须与真实900一致。所有原失败JOBS/log/FAILURE/partial900proof保持并绑定，不能把已有失败proof当PASS。

六个source/protocol文件：metadata.py、runtime.py、run.py、PREREG.json、README.zh-CN.md、staticcheck.py。只有新scope可写小INPUT_BINDINGS原bytes副本和600750VERIFY原bytes副本，其内PT和条件数组路径仍指原scope；不复制三套PT或大NPZ。新SOURCE_LEDGER独立绑定wrapper，原strictcommon所有科学函数对象、model/scan/data/learning/sources来源与ROOT/configsource始终ORIGINAL，不克隆contextlib或重绑定函数globals。

新鲜900阶段普通模块加载原verify_saved完整AST，唯一登记加强：原common.load_checkpoint(...final=False)→final=True，所有20actualX/12TF/12conditional循环和数学保持原样，只有该module.ROOT改新scope供INPUT读取及输出。common.load_checkpoint的临时观察wrapper调用原strictloader一次，返回完全相同model/payload对象；观察已验证payload的step900/8Adam900/identity/oldweights和curve原601..900日志基础字段逐行一致，允许checkpoint附加metrics_after_update。原anchor张量、RNG、矩形状等由同一原strictloader完整校验，不额外load一次900，不跨进程缓存model。该进程对optimizer step/zero_grad/backward设禁止陷阱，原验证前后weights digest相同/无grad；观察不改变模型或payload。

只有新900 fresh81checks全部通过，才在新TRAIN/RESULT写kind SAVED900_QUALIFIED_NO_NEW_UPDATES。其中updates900/new_updates300表示原训练已完成的更新，qualification_new_updates=0；显式引原失败与原300条日志，original_pipeline_success_claim=false。然后普通模块import原evaluate.py，只改module.ROOT新路径，读新600/750/900proof和新RESULT，strictload原900；原data.prepare仍读原INPUT。接着原audit_eval只ROOT新写新EVALUATION_VERIFY。原source/science/precision/cache均不变。

独立三阶段：FRESH900_SAVED_AUDIT / SAVED900_EVALUATION / INDEPENDENT_SAVED_METRICS。父实际125712 CPU1，整个新postrun共享3600秒硬截止，包括metadata及成功输出记账；RSS1800MiB，父进程与stage周期信号检查、剩余时间subprocess processgroup timeoutkill，无自动延长/重试。NN调用仅原fresh/eval科学审核，不能冒0；本工具训练/optimizer/gradient/Core/物理全0。成功写独立STAGE_VERIFICATION明确original_pipeline_all3exit0=false，随后回读所有success输出SHA并写RESOURCE_COMPLETION。任一FAILED/124/资源或proof失败不形成资格，保留原失败和新partial。

即使本工具成功，也**不满足原bodyprepare所要求的原三阶段exit0**。以后若接body须独立metadata兼容附录，明确postrunmode与原失败；当前不偷偷拿新JOBS冒原JOBS，也不宣body/brainneck PASS。旧accepted789f及所有失败保持。

子仅STD源码/JSON/ZIPheaders/opaqueSHA，不导NumPy/Torch、不读PT/NPZpayload/运行NN/Core/physics。STDpreview只核已完成600750，不读取正在追加的训练日志，原RUNNING时返回PENDING_ORIGINAL_RUNNING/qualified=false，不尝试数值资格。FIRSTREADY后代码冻结不改。

仅在未来真实准入成立、父资源独占且ACTUAL替原900实际64hex后运行一次：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T200608Z/postrun_saved900_qualification_ONLYCODE/run.py --execute-parent-postrun-saved900 --sha256 ACTUAL
```

只读STD源检查命令：

```bash
/usr/bin/python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T200608Z/postrun_saved900_qualification_ONLYCODE/staticcheck.py
```

范围只是保存checkpoint数值/存储/指标资格，BodyANNcomponent_NOTnewbrainneckdeployment；不新训练、不重复物理，不改主tracker。完整人工脑→realDN2queue→NNdecoder2/full432未部署不可省。
