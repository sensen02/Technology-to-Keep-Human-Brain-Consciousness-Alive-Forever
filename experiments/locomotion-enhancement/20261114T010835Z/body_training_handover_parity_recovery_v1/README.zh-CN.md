# 身体—训练源交接诊断：仅metadata schema恢复

新ROOT：`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/body_training_handover_parity_recovery_v1`。

原scope `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/body_training_handover_parity` 的114实际exit1，错误在metadata._source读取旧walk RESULT不存在actual_ms，尚未导入NumPy或读取数组。原scope全部源、CODECHECK和EXECUTION.log字节保持不变并由新proof绑定。错误是metadata模板假设，不是模型或身体失败原因；95四行为FAIL不改。

本次先实际STD JSON读四source RESULT/VERIFY全部keys、采样artifact/计数、initial和IDENTITY；独立header/SHA审核确认：stop/L/R teacher actual_ms15400、actual_native_steps1232000、teacher_decisions2880、trace.samples15400、nativeactual1232000/post768000；walk旧完整336脑颈源RESULT没有actual_ms或actual_native_steps，实际为physical_steps1232000、neural_records3080，其trace.samples15400与NPYtime.shape15400/nativeactual1232000构成时窗证据。不能给旧JSON伪补字段。

teacher sample artifact为teacher_samples.rows2880；walk为neural_decisions.rows3080，source旧motor hidden_before/after256，不把原state_total336误当这两个数组宽度。teacher hidden未记录。教师INITIAL保存7机械数组；walkINITIAL只qpos/qvel/head几何，其他机械字段NOT_RECORDED。walkIDENTITY有468compiled_arrays，但无top-levelactuator/physics/files；原数组分析对此已有缺录处理，不捏造新wrapper或把scene/group差异当物理原因。

最小修复：新metadata按固定SHA单层compile原metadata，只把原第103行result['actual_ms']读替换成_source_window_length(result,entry,paths)，按实际kind查上述计数/traceheader/native证据；其余qualify逻辑原样。追加source_contract记录真实RESULT/VERIFY/INITIAL/IDENTITY keys、count字段缺录与各文件SHA。新analyze按固定SHA编译原完整analyze AST到新namespace，仅进口新metadata，全部数组计算/索引/门槛/输出解释不改；避免克隆函数丢globals。

FIRSTFINALREADY前必须实际调用新metadata.qualify完成全部四95cases、9JOBS、各42审计、cal/EVAL/CONDITIONAL/source/data/proof哈希整链，只标准库不导NumPy/Torch、不读NPZpayload。证明保存STD_METADATA_QUALIFICATION.json，静态检查与实际metadata资格分开；不能再仅凭AST没找到错误就预称真实输入可跑。该STD检查不执行数组诊断、不预测parity结果。

父后续唯一数组分析仍≤90秒/RSS1500MiB，NumPy只读保存数组，0NN/PTload/Core/物理/训练。保持原teacher960与NN1160首样本、pretrace5799/firstupdates5800..5804/post5804；对比warm X/机械prefix、首新NN与EVAL/CONDITIONAL600保存参考。walk真实source task[1,raw_turn]与body[1,0]不同则NOT_CONTROLLED_FOR_COMMAND/nullinterfaceparity；teacherY只是标签，缺录old256不重算。初始一致不能证明后续闭环/泛化或行为PASS。原研究gates和失败完整保留。

父视频串行资源释放后准确命令：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/body_training_handover_parity_recovery_v1/analyze.py --execute-parent-saved-handover-analysis
```

标准库静态命令：

```bash
/usr/bin/python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/body_training_handover_parity_recovery_v1/static_check.py
```

新source首次最终冻结后不再编辑，不改旧cal/body/weights/tracker，不重复物理。完整人工brainNN→realDN2queue→NNdecoder2/full432仍未部署；范围BodyANNcomponent_NOTnewbrainneckdeployment。
