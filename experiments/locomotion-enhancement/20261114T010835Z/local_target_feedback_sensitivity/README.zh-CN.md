# 局部目标反馈敏感性诊断：ONLYCODE，未执行数值

目的仅是检验 new600 实际 stop/walk 更差是否涉及局部 target 自反馈。最终 batch 数值 exact 不等于物理行为通过；原 stop/walk/左右转行为结果与全部失败保留。本工具不调整网络、训练轮次、source、旧 VERIFY 或容差，不产生任何控制器部署。

本轮代码准备只读源码、JSON 和真实 NPZ 的 ZIP/NPY 头；不导入 Torch/NumPy，不读取数组值或 PT payload。纯标准库 staticcheck 可运行；数值入口必须由父在 body 九阶段完成、四个 fresh VERIFY 通过且资源独占后明确执行。body 行为 FAIL 不阻止本次诊断，但完整性 FAIL、数据缺失或选中 stop/walk 不完整会阻止执行。

模型接口已按原源码核对：`m.step_details(obs[B,238], command[B,2], state128/64/64/96)` 返回 mean、after352 tuple、residual；old 参考是 `m.old.step(obs, m.legacy_projection(command), state[:3])`。没有猜测不存在的 `_old_command` API，也不通过 autograd 测量被 no_grad/detach 切断的旧 prior 路径。函数源来自冻结 local_mean_corrective_calibration/model.py 与 combined_turning/model.py；严格模型加载由 v4 common 完成。

六个固定 preaction 点为 stop/walk 各 ANN+.5、4.8、9.0秒：physical_step_ms=6300/10600/14800，神经 row1260/2120/2960，preaction trace row6299/10599/14799，postaction trace row6304/10604/14804。使用 actual X238、hidden_before352、command2；target/qactual 来自 trace[physical_ms−1]，qpos 地址来自 CAL.INPUT_BINDINGS normalizer_mapping。q0/halfspan 与 BODY.IDENTITY compiled jnt_range/key_qpos 的 shape/dtype/SHA 以及 actual bounds/rate 匹配，全程不构建 Core。

每点先要求重编码 X42:84、126:168 与存档逐位相同；实际 public mean/action/residual/after352、old直接均值及五次1ms积分postT基线与存档≤1e-12，old256逐位exact。任一失败停止并保留失败，不减点、不改 epsilon、不切到近似 API。

之后固定该点其余 physical X、incoming 旧256/新96以及独立 command，对42个 target 坐标分别±1e-4、±2e-4 native rad 做中心差分。每次都用真实 target−qactual、target−q0 的 clip[-10,10]→float32→float64 编码和实际 public m.step_details，旧参考另行调用实际 m.old.step。积分严格沿用 action*2→clip±2→actual max_rate→五次1ms加法/bounds clip；此处只是重放局部 map，不是新动作控制器。

预算设计明确选择最小必需版，**不做138维target+newmemory完整map**。完整成功计数为 public step_details 1014 次、显式 old.step 1014 次、public内部 old.step 1014 次；顶层神经评估2028次，含嵌套model step调用3042次。每次均B1，全部no_grad，0训练、0optimizer、0梯度、0Core、0物理。整个严格加载、数据身份、有限差分和输出均计入300秒/1800MiB/单CPU限制；actual时间未知，超预算保留失败，不自动扩预算或缩方案。

保存42×42 dmean/dT、daction/dT、dpostT/dT、96×42 dafter96/dT，以及 old mean/postT参考与量化 -gain expectation。raw特征值real/imag、谱半径、Frobenius/operator范数、正对角计数、边界跨越逐项保存。old FD 与实际 float32 offset 差分的 -gain 预期需≤1e-8；这一FD sanity门不替换存档基线1e-12门。未clip平滑参考 -gain/halfspan 另存，不能把它冒充实际量化导数。

amplification flag 只在两epsilon的 dpostT/dT 谱半径都≥1.0001、rho差≤0.001、矩阵Frobenius相对差≤.05时标记；稳健正mean对角通道要求两epsilon同通道都≥1e-4。raw结果即使不一致也保留。这些是固定身体/固定incoming状态的局部有限差分，不证明138维耦合系统或fullbody稳定性，也不能单独证明行为失败唯一原因。

冻结支持代码复用：

`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/final_batch_parity_audit/audit.py`

其SHA固定为79cb7ddf4b9bcb3849af53265d59fa0d9774fcb082a5d630f1798fb43c2cee05，仅直接复用已经验证的标准库 final_qualification/v4_context 原函数对象，不克隆装饰器。它负责显式 v4 adapter/sources/common/model/learning/scan/data/utils alias/path 上下文并 finally 恢复。另核最终 batch PASS、CAL全资格、BODY九阶段/四VERIFY/summary/artifacts/compiled identity。动态body JOBS只在实际执行资格门读入，不在本轮编写时推定完成或编入固定数值结论。

父唯一实际命令（必须managed、独占执行并collect；本子不执行）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/local_target_feedback_sensitivity/diagnose.py --checkpoint /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_order_repair_v4/TRAIN/checkpoint_000600.pt --sha256 41fa1e2232f5143fec0a3ebb5d506c819af0ba084b3a05041e829fd1477b906b --execute-parent-local-target-sensitivity
```

本工具在创建watchdog及导入Torch之前即设置单CPU affinity，使所有后续线程继承同一mask；数值前另枚举所有live thread的affinity并要求完全同CPU。intra/inter-op=1及OMP/BLAS/MKL=1，有300秒SIGALRM和1800MiB RSS watchdog。父可额外外部timeout兜底但不扩预算。独占输出：

`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/local_target_feedback_sensitivity/RUN_41fa1e2232f5143fec0a3ebb5d506c819af0ba084b3a05041e829fd1477b906b/`

目录已存在拒绝覆盖。成功/失败均保存 RESULT.json、ROWS.json、SOURCE_LEDGER.json、中文 REPORT；完整成功另保存 FD_ARRAYS.npz。最终 RESOURCE_COMPLETION.json 在 RESULT 写完后记录 wall/RSS 并绑定 RESULT/ROWS/LEDGER/FD_ARRAYS/REPORT 的 SHA；watchdog/alarm保持至全部成功输出后的最后资源门。父仅可在脚本exit0、RESULT status=COMPLETED且diagnostic_integrity_pass=true、RESOURCE_COMPLETION.pass=true、全部artifact SHA一致并且无FINALIZATION_FAILURE.json时接纳本次诊断。异常分支强制清除pass，输出/资源信号失败也返回非0；预算信号后的最小失败记账可结束写盘，但永不授权科学完成。RESULT 单独的 COMPLETED 不能作最终准入，更不等于运动成功或local稳定性PASS。

源文件、checkpoint、weights、RNG、实际bodyproof/archives前后身份均核对；另要求BODY INPUT_BINDINGS.source_hashes为最终summary/四RESULT sourceledger的同身份子集，两selected IDENTITY.config.source_hashes也必须一致，再核对这些保存的源SHA与当前实际字节，不能用临时现采before/after掩盖事前源漂移。NN调用数透明记录。

纯标准库静态命令：

```bash
/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/local_target_feedback_sensitivity/staticcheck.py
```

仅检查AST/递归globals/实际API名字/六点两epsilon和理论调用计数、strictfinalTrue/no_grad/无optimizer/autograd、actualZIP头，以及内存body metadata正例（允许行为FAIL）、九个拒绝例和异常状态／资源completion拒绝例；不读取实际数组或权重。ONLYCODE静态通过与实际诊断结果分开记录。
